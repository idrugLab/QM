# -*- coding: utf-8 -*-

import ase

import torch
from torch import nn
from torch.nn import Embedding, Linear, ModuleList

from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock, ShiftedSoftplus


class FeatureAttention(nn.Module):
    def __init__(self, hidden_channels, squeeze='sum', reduction=2):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // reduction)
        self.lin2 = nn.Linear(hidden_channels // reduction, hidden_channels)
        self.squeeze = squeeze

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, batch, size=None):
        result = scatter(x, batch, dim=0, dim_size=size, reduce=self.squeeze)
        out = self.lin2(torch.relu(self.lin1(result)))
        y = torch.sigmoid(out)
        y = y[batch]
        return x * y


class SchNet(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50,
                 cutoff=10.0, max_num_neighbors=32, readout='add', dipole=False, mean=None,
                 std=None, atomref=None, squeeze='sum', reduction=2, location=None):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.attentions = ModuleList()
        for _ in range(num_interactions):
            block = FeatureAttention(hidden_channels, squeeze, reduction)
            self.attentions.append(block)
        self.location = location

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        for attention in self.attentions:
            attention.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, data):
        z = data.z
        pos = data.pos
        batch = data.batch
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        if self.location is None:
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

        elif self.location == 'standard':
            for interaction, attention in zip(self.interactions, self.attentions):
                h = h + attention(interaction(h, edge_index, edge_weight, edge_attr), batch)

        elif self.location == 'pre':
            for interaction, attention in zip(self.interactions, self.attentions):
                h = h + interaction(attention(h, batch), edge_index, edge_weight, edge_attr)

        elif self.location == 'post':
            for interaction, attention in zip(self.interactions, self.attentions):
                h = h + interaction(h, edge_index, edge_weight, edge_attr)
                h = attention(h, batch)

        elif self.location == 'identity':
            for interaction, attention in zip(self.interactions, self.attentions):
                h = interaction(h, edge_index, edge_weight, edge_attr) + attention(h, batch)

        else:
            raise ValueError(f'Location "{self.location}" is not supported yet.')

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


def build_model(arg):
    model = SchNet(hidden_channels=arg.hidden_channels,
                   num_filters=arg.num_filters,
                   num_interactions=arg.num_interactions,
                   num_gaussians=arg.num_gaussians,
                   cutoff=arg.cutoff,
                   squeeze=arg.squeeze,
                   reduction=arg.reduction,
                   location=arg.location,)

    return model

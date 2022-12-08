# -*- coding: utf-8 -*-

import os
import argparse
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from utils import seed_set, ThreeDEvaluator
from dataset import QM93D
from model import build_model


class run:
    def __init__(self):
        pass

    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=500,
            batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay=0.99, weight_decay=0, energy_and_force=False,
            p=100, save_dir='', log_dir=''):

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')

        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(1, epochs + 1):
            print("=====Epoch {}".format(epoch), flush=True)

            print('Training...', flush=True)
            train_mae = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device)

            print('Evaluating...', flush=True)
            valid_mae = self.val(model, valid_loader, energy_and_force, p, evaluation, device)

            print('Testing...', flush=True)
            test_mae = self.val(model, test_loader, energy_and_force, p, evaluation, device)

            print()
            print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})

            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)

            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid,
                                  'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            scheduler.step()

        print(f'Best validation MAE so far: {best_valid}')
        print(f'Test MAE when got best validation result: {best_test}')

        if log_dir != '':
            writer.close()

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device):
        model.train()
        loss_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out), create_graph=True,
                              retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss + p * f_loss
            else:
                loss = loss_func(out, batch_data.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        return loss_accum / (step + 1)

    def val(self, model, data_loader, energy_and_force, p, evaluation, device):
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)

        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out), create_graph=True,
                              retain_graph=True)[0]
                preds_force = torch.cat([preds_force, force.detach_()], dim=0)
                targets_force = torch.cat([targets_force, batch_data.force], dim=0)
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

        input_dict = {"y_true": targets, "y_pred": preds}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            print({'Energy MAE': energy_mae, 'Force MAE': force_mae})
            return energy_mae + p * force_mae

        return evaluation.eval(input_dict)['mae']


def parse_args():
    parser = argparse.ArgumentParser(description="Codes for SchNet")

    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=32)
    parser.add_argument('--seed', type=int, help="Random seed", default=1)

    parser.add_argument('--hidden_channels', type=int, help="Hidden embedding size", default=128)
    parser.add_argument('--num_filters', type=int, help="The number of filters to use", default=128)
    parser.add_argument('--num_interactions', type=int, help="The number of interaction blocks", default=5)
    parser.add_argument('--num_gaussians', type=int, help="The number of gaussians", default=50)
    parser.add_argument('--cutoff', type=float, help="Cutoff distance for interatomic interactions", default=10.0)
    parser.add_argument('--squeeze', type=str, help="The squeeze operator of the FA block", default='sum')
    parser.add_argument('--reduction', type=int, help="The reduction of the FA block", default=2)
    parser.add_argument('--location', type=str, help="The location of the FA block", default=None)

    parser.add_argument('--path', type=str, help="Directory where the dataset should be saved", default='./data/')
    parser.add_argument('--target', type=str, help="The task's name", default='mu')
    # 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    parser.add_argument('--save_dir', type=str, help="The directory of ckpt output", default='./output')
    parser.add_argument('--log_dir', type=str, help="The directory of logger output", default='./output')
    parser.add_argument('--tag', type=str, help='The tag of experiment', default='qm9_default')
    parser.add_argument('--epoch', type=int, help="Max epoch for training", default=300)
    parser.add_argument('--lr', type=int, help="Learning rate", default=1e-3)
    parser.add_argument('--l2', type=int, help="Weight decay", default=1e-4)

    arg = parser.parse_args()
    arg.log_dir = os.path.join(arg.log_dir, arg.tag)
    arg.save_dir = os.path.join(arg.save_dir, arg.tag)

    return arg


if __name__ == "__main__":
    seed_set(42)
    arg = parse_args()

    # print device mode
    if torch.cuda.is_available():
        print('device: GPU')
    else:
        print('device: CPU')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    dataset = QM93D(root='data/')
    target = arg.target
    dataset.data.y = dataset.data[target]
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=arg.seed)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], \
                                                 dataset[split_idx['test']]
    print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

    model = build_model(arg)
    loss_func = torch.nn.MSELoss()
    evaluation = ThreeDEvaluator()

    # training
    run3d = run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func,
              evaluation, epochs=arg.epoch, batch_size=arg.batch_size, vt_batch_size=arg.batch_size,
              lr=arg.lr, lr_decay=0.99, weight_decay=arg.l2, save_dir=arg.save_dir, log_dir=arg.log_dir)

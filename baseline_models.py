from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhNet(nn.Module):
    def __init__(self, in_features, h_units):
        super(TanhNet, self).__init__()
        self.fc1 = nn.Linear(in_features, h_units)
        self.fc2 = nn.Linear(h_units, 10)
        self.in_features = in_features
        self.h_units = h_units

    def forward(self, input: torch.Tensor):
        flattened = input.view(-1, self.in_features)
        input = F.tanh(self.fc1(flattened))
        input = F.tanh(self.fc2(input))
        return input

    def name(self, dset_name: str):
        return "tanh_{}x{}_{}".format(
            self.in_features, self.h_units, dset_name)

    def store(self, dset_name: str, directory: Path):
        name = self.name(dset_name)
        fname = "{}.tch".format(name)
        torch.save(self, str(directory / fname))


class ReLUNet(nn.Module):
    def __init__(self, in_features, h_units):
        super(ReLUNet, self).__init__()
        self.fc1 = nn.Linear(in_features, h_units)
        self.fc2 = nn.Linear(h_units, 10)
        self.in_features = in_features
        self.h_units = h_units

    def forward(self, input: torch.Tensor):
        flattened = input.view(-1, self.in_features)
        input = F.relu(self.fc1(flattened))
        input = F.relu(self.fc2(input))
        return input

    def name(self, dset_name: str):
        return "relu_{}x{}_{}".format(
            self.in_features, self.h_units, dset_name)

    def store(self, dset_name: str, directory: Path):
        name = self.name(dset_name)
        fname = "{}.tch".format(name)
        torch.save(self, str(directory / fname))


class MaxoutNet(nn.Module):
    def __init__(self, in_features, h_units, out_features=10):
        super(MaxoutNet, self).__init__()
        self.fc1 = Maxout(in_features, h_units, 2)
        self.fc2 = Maxout(h_units, out_features, 2)
        self.in_features = in_features
        self.h_units = h_units

    def forward(self, input: torch.Tensor):
        flattened = input.view(-1, self.in_features)
        input = self.fc1(flattened)
        input = self.fc2(input)
        return input

    def name(self, dset_name: str):
        return "maxout_{}x{}_{}".format(
            self.in_features, self.h_units, dset_name)

    def store(self, dset_name: str, directory: Path):
        name = self.name(dset_name)
        fname = "{}.tch".format(name)
        torch.save(self, str(directory / fname))

# from https://github.com/pytorch/pytorch/issues/805


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        maxout, _i = out.view(*shape).max(max_dim)
        return maxout

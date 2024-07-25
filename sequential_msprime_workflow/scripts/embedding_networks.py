import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sbi.neural_nets.embedding_nets import *


class SymmetricLayer(nn.Module):
    """
    Layer that performs some permutation-invariant function along a
    specified axis of input data.

    The permuation invariant function can be any of max, mean, or sum
    """

    def __init__(self, axis, func="max"):
        super().__init__()
        self.axis = axis
        self.func = func

    def forward(self, x):
        if self.func == "max":
            return torch.max(x, dim=self.axis, keepdim=True)[0]
        elif self.func == "mean":
            return torch.mean(x, dim=self.axis, keepdim=True)
        elif self.func == "sum":
            return torch.sum(x, dim=self.axis, keepdim=True)
        else:
            raise ValueError("func must be one of 'max', 'mean', or 'sum'")


class ExchangeableCNN(nn.Module):
    """
    This implements the Exchangeable CNN or permuation-invariant CNN from:
        Chan et al. 2018, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7687905/

    which builds in the invariance of the haplotype matrices to permutations of the individuals

    If input features come from multiple populations that may differ in num_snps and/or
    num_individuals, then provide a list of tuples with each populations haplotype matrix
    shape in unmasked_x_shps. The forward pass will then mask out all padded values of -1
    which pad each haplotype matrix to the shape of the largest in the set
    """

    def __init__(self, latent_dim=5, unmasked_x_shps=None, channels=2, symmetric_func="max"):
        """
        :param latent_dim: The desired dimension of the final 1D output vector
            to be used as the embedded data for training
        :param unmasked_x_shps: This is the shapes of each populations feature matrix
            before being padded. Needs to be given if we have mutliple differently sized
            feature matrices for different populations
        :param channels: The number of channels in the input matrices. HaplotypeMatrices
            have 2 channels and BinnedHaplotypeMatrices have 1 channel
        :param symmetric_func: String denoting which symmetric function to use in our
            permutation invariant layers
        """
        super().__init__()
        self.sizes1 = (32, 64)
        self.sizes2 = (64,)
        self.cnn_kernel_size = (1, 5)
        self.activation = nn.ELU
        self.unmasked_x_shps = unmasked_x_shps
        feat_ext_inp_dim = 64 if unmasked_x_shps is None else 64 * len(unmasked_x_shps)
        cnn_layers = []
        for in_size, feature_size in zip([channels, *self.sizes1], self.sizes1):
            cnn_layers.append(
                nn.Conv2d(
                    in_size,
                    feature_size,
                    self.cnn_kernel_size,
                    stride=(1, 2),
                    bias=False,
                )
            )
            cnn_layers.append(self.activation())
            cnn_layers.append(nn.BatchNorm2d(num_features=feature_size))
        cnn_layers.append(SymmetricLayer(axis=2, func=symmetric_func))
        for feature_size in self.sizes2:
            cnn_layers.append(
                nn.Conv2d(
                    feature_size,
                    feature_size,
                    self.cnn_kernel_size,
                    stride=(1, 2),
                    bias=False,
                )
            )
            cnn_layers.append(self.activation())
            cnn_layers.append(nn.BatchNorm2d(num_features=feature_size))
        cnn_layers.append(SymmetricLayer(axis=3, func=symmetric_func))
        self.cnn = nn.Sequential(*cnn_layers)
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_ext_inp_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def __call__(self, x):
        # if unmasked_x_shps is not None this means we have mutliple populations and
        # thus could have padded values of -1 we want to make sure to choose a mask
        # that pulls out all values of the different populations feature matrices,
        # EXCEPT those that equal -1
        if self.unmasked_x_shps is not None and len(x.shape) == 5:
            xs = []
            batch_ndim = x.shape[0]
            for i, shape in enumerate(self.unmasked_x_shps):
                mask = x[:, i, :, :, :] != -1
                inds = torch.where(mask)
                x_ = x[:, i, :, :, :][inds].view(batch_ndim, *shape)
                xs.append(self.cnn(x_))
            x = torch.cat(xs, dim=-1)
            return self.feature_extractor(x)
        # Otherwise we know there are no padded values and can just run the
        # input data through the network
        return self.feature_extractor(self.cnn(x))

    def embedding(self, x):
        with torch.no_grad():
            if self.unmasked_x_shps is not None and len(x.shape) == 5:
                xs = []
                batch_ndim = x.shape[0]
                for i, shape in enumerate(self.unmasked_x_shps):
                    mask = x[:, i, :, :, :] != -1
                    inds = torch.where(mask)
                    x_ = x[:, i, :, :, :][inds].view(batch_ndim, *shape)
                    xs.append(self.cnn(x_))
                x = torch.cat(xs, dim=-1)
                return self.feature_extractor[:2](x)
            return self.feature_extractor[:2](self.cnn(x))


class SPIDNA(nn.Module):
    def __init__(self, num_output, num_block=7, num_feature=50, device='cuda', **kwargs):
        super(SPIDNA, self).__init__()
        self.num_output = num_output
        self.conv_pos = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_pos_bn = nn.BatchNorm2d(num_feature)
        self.conv_snp = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_snp_bn = nn.BatchNorm2d(num_feature)
        self.blocks = nn.ModuleList([SPIDNABlock(num_output, num_feature) for i in range(num_block)])
        self.device = device

    def forward(self, x):
        pos = x[:, 0, :].view(x.shape[0], 1, 1, -1)
        snp = x[:, 1:, :].unsqueeze(1)
        pos = F.relu(self.conv_pos_bn(self.conv_pos(pos))).expand(-1, -1, snp.size(2), -1)
        snp = F.relu(self.conv_snp_bn(self.conv_snp(snp)))
        x = torch.cat((pos, snp), 1)
        output = torch.zeros(x.size(0), self.num_output).to(self.device)
        for block in self.blocks:
            x, output = block(x, output)

        return output


class FlagelNN(nn.Module):
    def __init__(self, SNP_max, nindiv, num_params, kernel_size=2, pool_size=2, use_dropout=True):
        super(FlagelNN, self).__init__()
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv1d(in_channels=nindiv, out_channels=128, kernel_size=2)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2)
        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2)
        self.pool4 = nn.AvgPool1d(kernel_size=2)
        self.dropout4 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        # Dynamically compute the flattened output size
        conv_output_size = SNP_max
        for _ in range(4):
            conv_output_size = (conv_output_size - (kernel_size - 1)) // pool_size
        self.flattened_size = conv_output_size * 128

        self.dense_b2 = nn.Linear(SNP_max, 32)
        self.dropout_b2 = nn.Dropout(0.25)

        self.dense_final = nn.Linear(self.flattened_size + 32, 256)  # Concatenated size
        self.dropout_final = nn.Dropout(0.25)
        self.output_layer = nn.Linear(256, num_params)

    def forward(self, x1, x2):
        x1 = self.pool1(F.relu(self.conv1(x1)))
        if self.use_dropout:
            x1 = self.dropout1(x1)

        x1 = self.pool2(F.relu(self.conv2(x1)))
        if self.use_dropout:
            x1 = self.dropout2(x1)

        x1 = self.pool3(F.relu(self.conv3(x1)))
        if self.use_dropout:
            x1 = self.dropout3(x1)

        x1 = self.pool4(F.relu(self.conv4(x1)))
        if self.use_dropout:
            x1 = self.dropout4(x1)

        x1 = self.flatten(x1)

        x2 = F.relu(self.dense_b2(x2))
        if self.use_dropout:
            x2 = self.dropout_b2(x2)

        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.dense_final(x))
        if self.use_dropout:
            x = self.dropout_final(x)
        x = self.output_layer(x)

        return x


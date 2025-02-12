import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# TODO:
# - make this "exchangeable" by shuffling all columns but the last (requires handling packed_sequence separately)
class RNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, dropout=0.0):
        """
        :param input_size: the input size of the GRU layer, e.g. num_individuals*ploidy
            or num_individuals depending if the data is phased or not
        :param output_size: the output size of the network
        """
        super().__init__()
        self.rnn = nn.GRU(input_size, 84, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(168 * num_layers, 256), 
            nn.Dropout(dropout),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        _, hn = self.rnn(x) # (2 * layers, batch, 84)
        hn = hn.permute(1, 0, 2).reshape(hn.shape[1], -1) # (batch, 2 * layers * 84)
        return self.mlp(hn)


# TODO: SBI has a built-in exchangeable layer, why not use this?
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
            # TODO: why is this indexed?
            return torch.max(x, dim=self.axis, keepdim=True)[0]
        elif self.func == "mean":
            return torch.mean(x, dim=self.axis, keepdim=True)
        elif self.func == "sum":
            return torch.sum(x, dim=self.axis, keepdim=True)
        else:
            raise ValueError("func must be one of 'max', 'mean', or 'sum'")


# TODO cleanup: 
# - BUG: this isn't working if sample sizes aren't equal across pops
# - should work with a single pop given the ts_processor (make as general as possible)
# - let the number of channels/kernel sizes/etc be settable
# - remove the need to specify the unpadded input shapes, this can be figured out in forward
# - the logic in forward requires a batch dimension
# - could use the built-in symmetric layer from SBI
class ExchangeableCNN_IN(nn.Module):
    """
    This implements the Exchangeable CNN or permuation-invariant CNN from:
        Chan et al. 2018, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7687905/

    which builds in the invariance of the haplotype matrices to permutations of the individuals

    Main difference is that the first cnn has wider kernel and stride to capture the long range LD.

    If input features come from multiple populations that may differ in num_snps and/or
    num_individuals, then provide a list of tuples with each populations haplotype matrix
    shape in unmasked_x_shps. The forward pass will then mask out all padded values of -1
    which pad each haplotype matrix to the shape of the largest in the set

    It has two cnn layers, followed by symmetric layer that pools over the individual axis and feature extractor (fully connected network).
    Each CNN layer has 2D convolution layer with kernel and stride height = 1, ELU activation, and Batch normalization layer.
    If the number of popultion is greater than one, the output of the first CNN layer is concatenated along the last axis.
    (same as pg-gan by Mathieson et al.)
    Then global pool make output dim (batch_size, outchannels2, 1, 1) and then pass to the feature extractor.
    """

    def __init__(self, output_dim=64, input_rows=None, input_cols=None, channels=2, symmetric_func="max"):
        """
        :param output_dim: The desired dimension of the final 1D output vector
            to be used as the embedded data for training
        :param input_rows: The number of rows (samples) per population genotype matrix
        :param input_cols: The number of cols (SNPs) per population genotype matrix
        :param channels: The number of channels in the input matrices. HaplotypeMatrices
            have 2 channels and BinnedHaplotypeMatrices have 1 channel
        :param symmetric_func: String denoting which symmetric function to use in our
            permutation invariant layers
        """
        super().__init__()
        self.outchannels1 = 32
        self.outchannels2 = 160
        self.kernel_size1 = (1, 5)
        self.kernel_size2 = (1, 5)
        self.stride1 = (1, 2)
        self.stride2 = (1, 2)

        self.activation = nn.ELU
        self.unmasked_x_shps = None
        if input_rows is not None and input_cols is not None:
            assert len(input_rows) == len(input_cols)
            self.unmasked_x_shps = [
                (channels, r, c) for r, c in zip(input_rows, input_cols)
            ]
        cnn_layers = []
        cnn_layers.append(nn.Conv2d(channels, self.outchannels1, self.kernel_size1, stride=self.stride1))
        cnn_layers.append(self.activation())
        cnn_layers.append(nn.BatchNorm2d(num_features=self.outchannels1, track_running_stats=False))
        cnn_layers.append(nn.Conv2d(self.outchannels1, self.outchannels2, self.kernel_size2, stride=self.stride2))
        cnn_layers.append(self.activation())
        cnn_layers.append(nn.BatchNorm2d(num_features=self.outchannels2, track_running_stats=False))

        self.cnn = nn.Sequential(*cnn_layers)
        self.symmetric = SymmetricLayer(axis=2, func=symmetric_func)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.outchannels2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
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
                xs.append(self.symmetric(self.cnn(x_)))
            x = torch.cat(xs, dim=-1)
            x = self.globalpool(x)
            return self.feature_extractor(x)
        # Otherwise we know there are no padded values and can just run the
        # input data through the network
        return self.feature_extractor(self.globalpool(self.symmetric(self.cnn(x))))
    
class SummaryStatisticsEmbedding(nn.Module):
    """
    Embed summary statistics of a tree sequence.
    This is simply an identity layer that takes in a tensor of summary statistics
    (e.g., SFS) and outputs the same tensor.

    For single population SFS: input shape is (num_samples + 1,)
    For joint SFS: input shape is (num_samples_pop1 + 1, num_samples_pop2 + 1)
    """
    def __init__(self, output_dim=None):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        # Ensure input is a torch tensor and flatten if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        return self.identity(x.reshape(x.shape[0], -1))

    def embedding(self, x):
        """
        Consistent with other embedding networks, provide an embedding method
        that returns the same output as forward() since this is an identity layer
        """
        with torch.no_grad():
            return self.forward(x)

class SPIDNA(nn.Module):
    """
    SPIDNA architecture

    Parameters
    ----------
    output_dim : int
        Dimension of the output feature vector
    num_block : int
        Number of SPIDNA blocks in the network
    num_feature : int
        Number of features in the convolutional layers
    **kwargs : dict
        Additional keyword arguments
    """
    def __init__(self, output_dim=64, num_block=3, num_feature=32, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.conv_pos = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_pos_bn = nn.BatchNorm2d(num_feature)
        self.conv_snp = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_snp_bn = nn.BatchNorm2d(num_feature)
        self.blocks = nn.ModuleList([SPIDNABlock(output_dim, num_feature) for i in range(num_block)])

    def forward(self, x):
        # Get device from input tensor
        device = x.device
        
        pos = x[:, 0, :].view(x.shape[0], 1, 1, -1)
        snp = x[:, 1:, :].unsqueeze(1)
        pos = F.relu(self.conv_pos_bn(self.conv_pos(pos))).expand(-1, -1, snp.size(2), -1)
        snp = F.relu(self.conv_snp_bn(self.conv_snp(snp)))
        x = torch.cat((pos, snp), 1)
        output = torch.zeros(x.size(0), self.output_dim, device=device)
        for block in self.blocks:
            x, output = block(x, output)
        return output

    def embedding(self, x):
        """
        Compute the embedding of input data without gradients.
        
        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            Input tensor containing position and SNP data
            
        Returns
        -------
        torch.Tensor
            Embedded representation of the input data
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.forward(x)

class SPIDNABlock(nn.Module):
    """
    A block in the SPIDNA network that processes features and updates the output.
    
    Parameters
    ----------
    output_dim : int
        Dimension of the output feature vector
    num_feature : int
        Number of features in the convolutional layers
    """
    def __init__(self, output_dim, num_feature):
        super().__init__()
        self.output_dim = output_dim
        self.phi = nn.Conv2d(num_feature * 2, num_feature, (1, 3))
        self.phi_bn = nn.BatchNorm2d(num_feature * 2)
        self.maxpool = nn.MaxPool2d((1, 2))
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x, output):
        x = self.phi(self.phi_bn(x))
        psi1 = torch.mean(x, 2, keepdim=True)
        psi = psi1
        current_output = self.fc(torch.mean(psi[:, :self.output_dim, :, :], 3).squeeze(2))
        output = output + current_output
        psi = psi.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, psi), 1)
        x = F.relu(self.maxpool(x))
        return x, output


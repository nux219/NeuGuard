import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
#         pass
        weight = self.linear.weight.data.cpu().numpy()
        percentile_value = np.percentile(np.abs(weight), q)
        # print('percentile_value: ', percentile_value)
        mask = self.mask
        new_mask = np.where(abs(weight) < percentile_value, 0.0, mask)
        self.linear.weight.data = torch.from_numpy(weight * new_mask).to(device).float()
        self.mask = new_mask
        # self.mask = torch.from_numpy(self.mask).to(device)
        
        new_mask = new_mask.flatten()
        num_parameters = new_mask.shape[0]
        num_nonzero_parameters = (new_mask != 0).sum()
        self.sparsity = 1. - float(num_nonzero_parameters) / num_parameters

    def prune_by_percentage_mod(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
#         pass
        weight = self.linear.weight.data.cpu().numpy()
        percentile_value = np.percentile(np.abs(weight), q)
        # print('percentile_value: ', percentile_value)
        mask = self.mask
        new_mask = np.where(abs(weight) > percentile_value, 0.0, mask)
        self.linear.weight.data = torch.from_numpy(weight * new_mask).to(device).float()
        self.mask = new_mask
        # self.mask = torch.from_numpy(self.mask).to(device)
        # new_mask = new_mask.flatten()
        # num_parameters = new_mask.shape[0]
        # num_nonzero_parameters = (new_mask != 0).sum()
        # self.sparsity = 1. - float(num_nonzero_parameters) / num_parameters

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        # pass

        weight = self.linear.weight.data.cpu().numpy()
        threshold_value = np.std(weight)*s

        # print('weight: ', weight.shape, weight.dtype)
        # print(threshold_value.dtype)
        # print('weight_transpose: ', weight_transpose.shape)
        mask = self.mask
        # mask = np.ones_like(weight_transpose)
        # self.mask = mask
        # print('mask: ', mask.shape)
        # print('self.mask: ', self.mask.shape)
        new_mask = np.where(abs(weight) < threshold_value, 0.0, mask)
        self.linear.weight.data = torch.from_numpy(weight * new_mask).to(device).float()
        # print(self.linear.weight.data.dtype)
        # print('bias: ',self.linear.bias.data.dtype)
        
        self.mask = new_mask
        # self.mask = torch.from_numpy(self.mask).to(device)
        # print(self.mask.dtype)
        
        new_mask = new_mask.flatten()
        num_parameters = new_mask.shape[0]
        num_nonzero_parameters = (new_mask != 0).sum()
        # print('num_parameters : %f, num_nonzero_parameters: %f'%(num_parameters, num_nonzero_parameters))
        self.sparsity = 1. - float(num_nonzero_parameters) / num_parameters
        # print('sparsity: %f' %(self.sparsity))


class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Expand and Transpose to match the dimension
        # self.mask = np.ones_like([out_channels, in_channels, kernel_size, kernel_size])
        # if self.mask is None:
        self.mask = np.ones_like(self.conv.weight.data)
        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """

        weight = self.conv.weight.data.cpu().numpy()
        percentile_value = np.percentile(np.abs(weight), q)
        # print('percentile_value: ', percentile_value)
        # print('weight: ', weight.shape)
        mask = np.ones_like(weight)
        # mask = self.mask
        # print('mask: ', mask.shape)
        new_mask = np.where(abs(weight) < percentile_value, 0.0, mask)
        self.conv.weight.data = torch.from_numpy(weight * new_mask).to(device).float()
        self.mask = new_mask
        # self.mask = torch.from_numpy(self.mask).to(device)

        new_mask = new_mask.flatten()
        num_parameters = new_mask.shape[0]
        num_nonzero_parameters = (new_mask != 0).sum()
        self.sparsity = 1. - float(num_nonzero_parameters) / num_parameters
        

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        # pass
        weight = self.conv.weight.data.cpu().numpy()
        # weight_transpose = np.transpose(weight,(1,0,2,3))
        threshold_value = np.std(weight)*s
        
        mask = np.ones_like(weight)
       
        new_mask = np.where(abs(weight) < threshold_value, 0.0, mask)
        self.conv.weight.data = torch.from_numpy(weight * new_mask).to(device).float()
        self.mask = new_mask
        # self.mask = torch.from_numpy(self.mask).to(device)
        
        new_mask = new_mask.flatten()
        num_parameters = new_mask.shape[0]
        num_nonzero_parameters = (new_mask != 0).sum()
        self.sparsity = 1. - float(num_nonzero_parameters) / num_parameters
        # print('sparsity: %f' %self.sparsity)

 # mask = np.ones_like(weight_transpose)
        # self.mask = mask
        # print('mask: ', mask.shape)
        # print('self.mask: ', self.mask.shape)



        # threshold_value = np.std(weight_transpose)*s

        # print('weight: ', weight.shape)
        # print('weight_transpose: ', weight_transpose.shape)
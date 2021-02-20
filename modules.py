import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# from torchvision.datasets import MNIST
# import torchvision.transforms as transforms


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)

        distribution_in = h_dim[-1]
        distribution_out = z_dim
        self.linear_mu = nn.Linear(distribution_in, distribution_out)
        self.linear_log_var = nn.Linear(distribution_in, distribution_out)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        mu = self.linear_mu(x)
        log_var = F.softplus(self.linear_log_var(x))
        return self.reparametrize(mu, log_var), mu, log_var

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)
        return z


class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class Autoencoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, init_std=0.001):
        super(Autoencoder, self).__init__()

        self.z_dim = z_dim

        self.encoder = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, h_dim=list(reversed(h_dim)), x_dim=x_dim)
        self.kl_divergence = 0
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=init_std)  # nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        (mu, log_var) = q_param

        KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return KL

    def forward(self, x, y=None):

        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        return self.decoder(z)


def sampleFromClass(ds, k):
    """
    :param ds: input dataset samples.
    :return: Train and Test datasets. Train has k samples per class. Test is the rest of the dataset samples.
    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in ds:
        c = label
        label_t = torch.Tensor([label])
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            train_data.append(torch.unsqueeze(data, 0))
            train_label.append(torch.unsqueeze(label_t, 0))
        else:
            test_data.append(torch.unsqueeze(data, 0))
            test_label.append(torch.unsqueeze(label_t, 0))
    train_data = torch.cat(train_data)

    train_label = torch.cat(train_label)
    test_data = torch.cat(test_data)
    test_label = torch.cat(test_label)
    trn = torch.utils.data.TensorDataset(train_data, train_label)
    tst = torch.utils.data.TensorDataset(test_data, test_label)
    return (trn, tst)


def class_counter_ds(dataset):
    """
    :param dataset: input dataset
    :return: dict: Counters[c]=k, such that class c has k samples in the dataset.
    """
    class_counts = {}
    for i in range(len(dataset)):
        if type(dataset[i][1]) != int:
            c = dataset[i][1].item()
        else:
            c = dataset[i][1]
        class_counts[c] = class_counts.get(c, 0) + 1
    return class_counts


def sampleRandFromClass(ds, k, seed=None, ret_loaders=False):
    """
    :param ds: input dataset samples.
    :param k: num of samples per class
    :param seed: random permutation seed for numpy. if =-1 using 'range(len(ds))' permutation.
    :param ret_loaders: if True return 'torch.utils.data.TensorDataset' objects
    :return: Train and Test datasets
    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    if seed != None and seed != -1:
        # torch.manual_seed(seed)
        np.random.seed(seed)
    perm_ind = np.random.permutation(len(ds))  # torch.randperm(len(ds)).numpy()
    if seed == -1:
        perm_ind = range(len(ds))
    for i in perm_ind:
        data, label = ds[i]
        c = label
        label_t = torch.Tensor([label])
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            train_data.append(torch.unsqueeze(data, 0))
            train_label.append(torch.unsqueeze(label_t, 0))
        else:
            test_data.append(torch.unsqueeze(data, 0))
            test_label.append(torch.unsqueeze(label_t, 0))
    train_data = torch.cat(train_data)

    train_label = torch.cat(train_label)
    test_data = torch.cat(test_data)
    test_label = torch.cat(test_label)
    if ret_loaders:
        trn = torch.utils.data.TensorDataset(train_data, train_label)
        tst = torch.utils.data.TensorDataset(test_data, test_label)
        return (trn, tst)
    else:
        return (train_data, train_label, test_data, test_label)


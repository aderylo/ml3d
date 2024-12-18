import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, input_dim, num_freqs):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

    def forward(self, x):
        """
        Compute the positional encoding of the input.
        Output = [x, sin(2^0 * x), cos(2^0 * x), sin(2^1 * x), cos(2^1 * x), ..., sin(2^num_freqs * x), cos(2^num_freqs * x]

        Args:
            x: (N, input_dim)

        Returns:
            (N, input_dim * (num_freqs * 2 + 1))
        """
        # TODO: implement forward pass
        pass


class SDFField(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.Softplus(beta=100)
        self.embed_fn = PositionalEmbedding(3, 6)
        # TODO: Define the rest of the model

    def forward(self, x):
        """
        Args:
            x: (N, 3) input points
        Output:
            (N, 1 + latent_size) tensor. The first value is the SDF value and the rest are the latent code.
        """
        # TODO: implement forward pass
        # The forward pass should contain the following steps:
        # 1. Compute the positional encoding of the input
        # 2. Apply mlp layers. Add skip connections if needed.
        pass

    def get_sdf(self, x):
        """Get the SDF value only without the latent code"""
        sdf = self.forward(x)[:, :1]
        return sdf

    def gradient(self, x):
        """Compute the normal direction using the gradient of the SDF"""
        x.requires_grad_(True)
        y = self.get_sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class ColorField(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.embed_fn = PositionalEmbedding(3, 4)
        # TODO: Define the rest of the model

    def forward(self, xyz, normals, view_dirs, features):
        """
        Args:
            xyz: (N, 3) input points
            normals: (N, 3) input normals
            view_dirs: (N, 3) input view directions
            features: (N, C) input features
        Returns:
            (N, 3) color output
        """
        # TODO: implement forward pass
        # The forward pass should contain the following steps:
        # 1. Add positional encoding to the view_dirs
        # 2. Concatenate the input features
        # 3. Apply mlp layers
        # 4. Apply sigmoid on the color output
        pass


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=0.3):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)


def compute_psnr(x, y, mask=None):
    # TODO: implement PSNR computation
    pass

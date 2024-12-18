import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        def layer(in_features, out_features):
            return nn.Sequential(
                torch.nn.utils.weight_norm(
                    nn.Linear(in_features, out_features)
                ),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            )
        self.block1 = nn.Sequential(
            layer(latent_size + 3, 512),
            layer(512, 512),
            layer(512, 512),
            layer(512, latent_size + 3),
        )
        self.block2 = nn.Sequential(
            layer(2 * (latent_size + 3), 512),
            layer(512, 512),
            layer(512, 512),
            layer(512, 512),
            nn.Linear(512, 1),
        )


    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.block1(x_in)
        x = self.block2(torch.concat((x_in, x), dim=1))
        return x

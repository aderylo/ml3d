import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        # TODO Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        # TODO Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU


        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k

    def forward(self, x):
        b = x.shape[0]

        # TODO Pass input through layers, applying the same max operation as in PointNetEncoder
        # TODO No batch norm and relu after the last Linear layer
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)


        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super().__init__()

        # TODO Define convolution layers, batch norm layers, and ReLU
        self.conv1 = nn.Conv1d(3, 64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(64, 128, kernel_size=(1,))
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=(1,))

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)

        self.relu = nn.ReLU()
        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)

        # TODO: First layer: 3->64
        x = self.relu(self.bn1(self.conv1(x)))

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # TODO: Layers 2 and 3: 64->128, 128->1024
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = PointNetEncoder(return_point_features=False)
        # TODO Add Linear layers, batch norms, dropout with p=0.3, and ReLU
        # Batch Norms and ReLUs are used after all but the last layer
        # Dropout is used only directly after the second Linear layer
        # The last Linear layer reduces the number of feature channels to num_classes (=k in the architecture visualization)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.encoder(x)
        # TODO Pass output of encoder through your layers

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(return_point_features=True)
        # TODO: Define convolutions, batch norms, and ReLU


        self.conv1 = nn.Conv1d(1088, 512, kernel_size=(1,))
        self.conv2 = nn.Conv1d(512, 256, kernel_size=(1,))
        self.conv3 = nn.Conv1d(256, 128, kernel_size=(1,))
        self.conv4 = nn.Conv1d(128, num_classes, kernel_size=(1,))

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        # TODO: Pass x through all layers, no batch norm or ReLU after the last conv layer
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        return x
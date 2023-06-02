import torch
import torch_geometric

from torch.nn import Linear, Sequential, ReLU
from torch.nn.functional import elu
from torch_geometric.nn.conv import EdgeConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX


class EDGE(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(EDGE, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 8, 16, 16, 16, 32, 32, 32]
            pooling_outputs = 32
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.mlp1 = Sequential(Linear(2*n[0], n[1]),
                              ReLU(),
                              Linear(n[1], n[1]))
        self.conv1 = EdgeConv(self.mlp1)
        self.norm1 = BatchNorm(in_channels=n[1])
        self.mlp2 = Sequential(Linear(2*n[1], n[2]),
                              ReLU(),
                              Linear(n[2], n[2]))
        self.conv2 = EdgeConv(self.mlp2)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.mlp3 = Sequential(Linear(2*n[2], n[3]),
                              ReLU(),
                              Linear(n[3], n[3]))
        self.conv3 = EdgeConv(self.mlp3)
        self.norm3 = BatchNorm(in_channels=n[3])
        self.mlp4 = Sequential(Linear(2*n[3], n[4]),
                              ReLU(),
                              Linear(n[4], n[4]))
        self.conv4 = EdgeConv(self.mlp4)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.mlp5 = Sequential(Linear(2*n[4], n[5]),
                              ReLU(),
                              Linear(n[5], n[5]))
        self.conv5 = EdgeConv(self.mlp5)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.mlp6 = Sequential(Linear(2*n[5], n[6]),
                              ReLU(),
                              Linear(n[6], n[6]))
        self.conv6 = EdgeConv(self.mlp6)
        self.norm6 = BatchNorm(in_channels=n[6])
        self.mlp7 = Sequential(Linear(2*n[6], n[6]),
                              ReLU(),
                              Linear(n[7], n[7]))
        self.conv7 = EdgeConv(self.mlp7)
        self.norm7 = BatchNorm(in_channels=n[7])

        #self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.pool7 = MaxPoolingX(torch.div(input_shape[:2], 4, rounding_mode='floor'), size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.edge_index))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index))
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)
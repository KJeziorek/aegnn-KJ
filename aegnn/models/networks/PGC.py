import torch
import torch_geometric

from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import elu
from torch_geometric.nn.conv import PointGNNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX


class PGC(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(PGC, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 64, 64, 64, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.mlp_h1 = Linear(n[0], n[1])
        self.mlp_f1 = Linear(n[0], n[1])
        self.mlp_g1 = Linear(n[0], n[1])
        self.conv1 = PointGNNConv(self.mlp_h1, self.mlp_f1, self.mlp_g1)
        self.norm1 = BatchNorm(in_channels=n[1])
        
        self.mlp_h2 = Linear(n[1], n[2])
        self.mlp_f2 = Linear(n[1], n[2])
        self.mlp_g2 = Linear(n[1], n[2])
        self.conv2 = PointGNNConv(self.mlp_h2, self.mlp_f2, self.mlp_g2)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.mlp_h3 = Linear(n[2], n[3])
        self.mlp_f3 = Linear(n[2], n[3])
        self.mlp_g3 = Linear(n[2], n[3])
        self.conv3 = PointGNNConv(self.mlp_h3, self.mlp_f3, self.mlp_g3)
        self.norm3 = BatchNorm(in_channels=n[3])

        self.mlp_h4 = Linear(n[3], n[4])
        self.mlp_f4 = Linear(n[3], n[4])
        self.mlp_g4 = Linear(n[3], n[4])
        self.conv4 = PointGNNConv(self.mlp_h4, self.mlp_f4, self.mlp_g4)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.mlp_h5 = Linear(n[4], n[5])
        self.mlp_f5 = Linear(n[4], n[5])
        self.mlp_g5 = Linear(n[4], n[5])
        self.conv5 = PointGNNConv(self.mlp_h5, self.mlp_f5, self.mlp_g5)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.mlp_h6 = Linear(n[5], n[6])
        self.mlp_f6 = Linear(n[5], n[6])
        self.mlp_g6 = Linear(n[5], n[6])
        self.conv6 = PointGNNConv(self.mlp_h6, self.mlp_f6, self.mlp_g6)
        self.norm6 = BatchNorm(in_channels=n[6])

        self.mlp_h7 = Linear(n[6], n[7])
        self.mlp_f7 = Linear(n[6], n[7])
        self.mlp_g7 = Linear(n[6], n[7])
        self.conv7 = PointGNNConv(self.mlp_h7, self.mlp_f7, self.mlp_g7)
        self.norm7 = BatchNorm(in_channels=n[7])

        #self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.pool7 = MaxPoolingX(torch.div(input_shape[:2], 4, rounding_mode='floor'), size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.pos, data.edge_index))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.pos, data.edge_index))
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.pos, data.edge_index))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.pos, data.edge_index))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.pos, data.edge_index))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.pos, data.edge_index))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.pos, data.edge_index))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)
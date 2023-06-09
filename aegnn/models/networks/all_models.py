import torch
import torch_geometric

from torch.nn import Sequential, Linear, ReLU, Dropout
from torch.nn.functional import elu
from torch_geometric.nn.conv import PointNetConv, EdgeConv, SAGEConv, SplineConv, GCNConv, PointGNNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX

class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = GCNConv(in_channels, out_channels)
        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = elu(self.conv(x, edge_index, edge_attr))
        x = self.norm(x)
        return x
    
class EdgeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = Linear(in_channels * 2, out_channels)
        self.conv = EdgeConv(self.mlp)
        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, edge_index):
        x = elu(self.conv(x, edge_index))
        x = self.norm(x)
        return x

class SplineBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = SplineConv(in_channels, out_channels, dim=3, kernel_size=8, bias=False, root_weight=False)
        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = elu(self.conv(x, edge_index, edge_attr))
        x = self.norm(x)
        return x
       
class SageBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = SAGEConv(in_channels, out_channels, root_weight=False, bias=False)
        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, edge_index):
        x = elu(self.conv(x, edge_index))
        x = self.norm(x)
        return x
    
class PointNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mlp = Linear(in_channels + 3, out_channels)
        self.conv = PointNetConv(self.mlp)
        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, pos, edge_index):
        x = elu(self.conv(x, pos, edge_index))
        x = self.norm(x)
        return x

class PointGNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.mlp_h = Linear(in_channels, 3)
        self.mlp_f = Linear(4, out_channels)
        self.mlp_g = Linear(out_channels, out_channels)
        self.conv = PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g)
        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = elu(self.conv(x, edge_index, edge_attr))
        x = self.norm(x)
        return x
    
class GNNModel(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GNNModel, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            n = [1, 64, 64, 64, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.point1 = PointNetBlock(1, 8)
        self.point2 = PointNetBlock(8, 16)

        self.pool1 = MaxPooling(
            (4, 4), transform=Cartesian(norm=True, cat=False))
        
        self.point3 = PointNetBlock(16, 32)
        self.point4 = PointNetBlock(32, 32)
        self.point5 = PointNetBlock(32, 32)

        self.pool2 = MaxPooling(
            (4, 4), transform=Cartesian(norm=True, cat=False))
        
        self.point6 = PointNetBlock(32, 64)
        self.point7 = PointNetBlock(64, 64)
        self.point8 = PointNetBlock(64, 64)

        self.pool3 = MaxPooling(
            (4, 4), transform=Cartesian(norm=True, cat=False))
        
        self.point9 = PointNetBlock(64, 128)
        self.point10 = PointNetBlock(128, 128)
        self.point11 = PointNetBlock(128, 128)
        
        # self.pool4 = MaxPooling(
        #     (4, 4), transform=Cartesian(norm=True, cat=False))
        
        # self.point12 = PointNetBlock(128, 256)
        # self.point13 = PointNetBlock(256, 256)
        # self.point14 = PointNetBlock(256, 256)

        self.pool_out = MaxPoolingX(torch.div(input_shape[:2], 4, rounding_mode='floor'), size=16)
        self.drop_out = Dropout(0.5)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = self.point1(data.x, data.pos, data.edge_index)
        data.x = self.point2(data.x, data.pos, data.edge_index)

        data = self.pool1(data.x, pos=data.pos, batch=data.batch,
                          edge_index=data.edge_index, return_data_obj=True)
        
        data.x = self.point3(data.x, data.pos, data.edge_index)
        x_sc = data.x.clone()
        data.x = self.point4(data.x, data.pos, data.edge_index)
        data.x = self.point5(data.x, data.pos, data.edge_index)
        data.x = data.x + x_sc

        data = self.pool2(data.x, pos=data.pos, batch=data.batch,
                          edge_index=data.edge_index, return_data_obj=True)
        
        data.x = self.point6(data.x, data.pos, data.edge_index)
        x_sc = data.x.clone()
        data.x = self.point7(data.x, data.pos, data.edge_index)
        data.x = self.point8(data.x, data.pos, data.edge_index)
        data.x = data.x + x_sc

        data = self.pool3(data.x, pos=data.pos, batch=data.batch,
                          edge_index=data.edge_index, return_data_obj=True)
        
        data.x = self.point9(data.x, data.pos, data.edge_index)
        x_sc = data.x.clone()
        data.x = self.point10(data.x, data.pos, data.edge_index)
        data.x = self.point11(data.x, data.pos, data.edge_index)
        data.x = data.x + x_sc

        # data = self.pool4(data.x, pos=data.pos, batch=data.batch,
        #                   edge_index=data.edge_index, return_data_obj=True)
        
        # data.x = self.point12(data.x, data.pos, data.edge_index)
        # x_sc = data.x.clone()
        # data.x = self.point13(data.x, data.pos, data.edge_index)
        # data.x = self.point14(data.x, data.pos, data.edge_index)
        # data.x = data.x + x_sc

        x = self.pool_out(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        x = self.drop_out(x)
        return self.fc(x)
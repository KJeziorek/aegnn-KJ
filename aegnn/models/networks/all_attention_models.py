import torch
import torch_geometric

from torch.nn import Linear, Sequential
from torch.nn.functional import elu
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import PointTransformerConv, TransformerConv, GATv2Conv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX

class GATv2Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()

        self.conv = GATv2Conv(in_channels, out_channels, heads=heads, bias=False, edge_dim=3, dropout=0.2)
        self.norm = BatchNorm(in_channels=out_channels*heads)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x)
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x)
        return x


class PointTransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pos_nn = Sequential(Linear(3, 64),
                                 Linear(64, out_channels))

        self.attn_nn = Sequential(Linear(out_channels, 64),
                                  Linear(64, out_channels))

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

        self.norm = BatchNorm(in_channels=out_channels)

    def forward(self, x, pos, edge_index):
        x = elu(self.transformer(x, pos, edge_index))
        x = self.norm(x)
        return x


class AttentionModel(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(AttentionModel, self).__init__()
        assert len(
            input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        pooling_outputs = 32

        self.conv1 = PointTransformerBlock(1, 32)

        self.pool2 = MaxPoolingX(
            torch.div(input_shape[:2], 4, rounding_mode='floor'), size=16)
        self.fc = Linear(pooling_outputs * 16,
                         out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = self.lin(data.x)
        data.x = self.conv1(data.x, data.pos, data.edge_index)

        x = self.pool2(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

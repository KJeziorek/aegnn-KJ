from .graph_res import GraphRes
from .graph_wen import GraphWen

from .all_models import GNNModel
from .all_attention_models import AttentionModel
#from .PGC import PGC  # only for PyTorch Geometric >= 2.3.0
################################################################################################
# Access functions #############################################################################
################################################################################################
import torch


def by_name(name: str) -> torch.nn.Module.__class__:
    if name == "graph_res":
        return GraphRes
    elif name == "graph_wen":
        return GraphWen
    elif name == "all":
        return GNNModel
    elif name == 'attention':
        return AttentionModel
    else:
        raise NotImplementedError(f"Network {name} is not implemented!")

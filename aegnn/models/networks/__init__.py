from .graph_res import GraphRes
from .graph_wen import GraphWen

from .TC import TC
from .EDGE import EDGE
from .PTC import PTC
from .SAGE import SAGE
from .PNC import PNC
from .GATv2 import GATv2
from .PGC import PGC
################################################################################################
# Access functions #############################################################################
################################################################################################
import torch


def by_name(name: str) -> torch.nn.Module.__class__:
    if name == "graph_res":
        return GraphRes
    elif name == "graph_wen":
        return GraphWen
    elif name == 'tc':
        return TC
    elif name == 'edge':
        return EDGE
    elif name == 'ptc':
        return PTC
    elif name == 'sage':
        return SAGE
    elif name == 'pnc':
        return PNC
    elif name == 'gat':
        return GATv2
    elif name == 'pgc':
        return PGC
    else:
        raise NotImplementedError(f"Network {name} is not implemented!")

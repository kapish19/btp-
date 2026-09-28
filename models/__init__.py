from .vision_transformer import ViT
try:
    from .GFNet import GFNet
except ImportError:
    GFNet = None
from .CLIP import Model as Clip
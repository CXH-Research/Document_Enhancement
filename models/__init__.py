from .BinFormer import BinFormer
from .DEGAN import DEGAN
from .DeepOtsu import DeepOtsu
from .DocEnTR import DocEnTR
from .DocNLC import DocNLC
from .DocProj import DocProj
from .DocTr import DocTr
from .GAN_HTR import GAN_HTR
from .GCDRNet import GCDRNet
from .SAE import SAE
from .SauvolaNet import SauvolaNet
from .TextDIAE import TextDIAE
from .UDoc_GAN import UDoc_GAN
from .dplinknet import DPLinkNet34
from .illtrtemplate import illtrtemplate
from .DocTrPP import DocTrPP
from .DocRes import DocRes
from .NAFNet import NAFNet

model_registry = {
    "DEGAN": DEGAN,
    "DeepOtsu": DeepOtsu,
    "DocNLC": DocNLC,
    "DocProj": DocProj,
    "GAN_HTR": GAN_HTR,
    "GCDRNet": GCDRNet,
    "SAE": SAE,
    "SauvolaNet": SauvolaNet,
    "UDoc_GAN": UDoc_GAN,
    "DocRes": DocRes,
    "NAFNet": NAFNet,
}
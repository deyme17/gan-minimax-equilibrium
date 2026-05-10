from .bce_loss import BCEGANLoss
from .wasserstein_loss import WassersteinLoss
from .hinge_loss import HingeLoss
from utils import Registry

LOSSES = Registry()

LOSSES.register("BCE")(BCEGANLoss)
LOSSES.register("Wasserstein")(WassersteinLoss)
LOSSES.register("Hinge")(HingeLoss)
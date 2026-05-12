from utils import Registry
from .discriminator import Discriminator
from .generator import BaselineGenerator, AdvancedGenerator

MODELS = Registry()

MODELS.register("D")(Discriminator)
MODELS.register("baseline_G")(BaselineGenerator)
MODELS.register("G")(AdvancedGenerator)
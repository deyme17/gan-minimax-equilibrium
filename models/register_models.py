from utils import Registry
from .discriminator import Discriminator, WGANCritic
from .generator import BaselineGenerator, AdvancedGenerator, WGANAdvancedGenerator

MODELS = Registry()

MODELS.register("D")(Discriminator)
MODELS.register("C")(WGANCritic)
MODELS.register("baseline_G")(BaselineGenerator)
MODELS.register("G")(AdvancedGenerator)
MODELS.register("WG")(WGANAdvancedGenerator)
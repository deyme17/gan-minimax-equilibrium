import pytest
import torch

from .generator import BaselineGenerator, AdvancedGenerator
from .discriminator import Discriminator


# fixtures

@pytest.fixture(scope="module")
def baseline_g():
    return BaselineGenerator(n_z=128, image_size=512, image_channels=3,
                             base_channels=512, min_channels=16).eval()

@pytest.fixture(scope="module")
def advanced_g():
    return AdvancedGenerator(n_z=128, image_size=512, image_channels=3,
                             base_channels=512, min_channels=16).eval()

@pytest.fixture(scope="module")
def discriminator():
    return Discriminator(base_channels=64, patch=False).eval()

@pytest.fixture(scope="module")
def discriminator_patch():
    return Discriminator(base_channels=64, patch=True).eval()

@pytest.fixture(scope="module")
def dataloader():
    from transforms import get_transform
    from utils import Config
    from dataset import get_dataloader
    cfg = Config()
    return get_dataloader(cfg, get_transform(cfg.input_dim)), cfg


# generator

class TestBaselineGenerator:

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape(self, baseline_g, batch_size):
        z = torch.randn(batch_size, 128)
        with torch.no_grad():
            out = baseline_g(z)
        assert out.shape == (batch_size, 3, 512, 512)

    def test_output_range(self, baseline_g):
        z = torch.randn(2, 128)
        with torch.no_grad():
            out = baseline_g(z)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_nan(self, baseline_g):
        z = torch.randn(2, 128)
        with torch.no_grad():
            out = baseline_g(z)
        assert torch.isfinite(out).all()

    def test_backward(self):
        G = BaselineGenerator()
        z = torch.randn(2, 128)
        G(z).mean().backward()
        assert any(p.grad is not None for p in G.parameters())


class TestAdvancedGenerator:

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape(self, advanced_g, batch_size):
        z = torch.randn(batch_size, 128)
        with torch.no_grad():
            out = advanced_g(z)
        assert out.shape == (batch_size, 3, 512, 512)

    def test_output_range(self, advanced_g):
        z = torch.randn(2, 128)
        with torch.no_grad():
            out = advanced_g(z)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_nan(self, advanced_g):
        z = torch.randn(2, 128)
        with torch.no_grad():
            out = advanced_g(z)
        assert torch.isfinite(out).all()

    def test_backward(self):
        G = AdvancedGenerator()
        z = torch.randn(2, 128)
        G(z).mean().backward()
        assert any(p.grad is not None for p in G.parameters())


# discriminator

class TestDiscriminator:

    def test_global_output_shape(self, discriminator):
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = discriminator(x)
        assert out.shape == (2, 1)

    def test_patch_output_shape(self, discriminator_patch):
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = discriminator_patch(x)
        assert out.shape[0] == 2 and out.shape[1] == 1  # (B, 1, H', W')

    @pytest.mark.parametrize("norm", ["batch", "instance", "none"])
    def test_norm_variants(self, norm):
        D = Discriminator(norm=norm, patch=False).eval()
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = D(x)
        assert out.shape == (2, 1)

    def test_spectral_norm(self):
        D = Discriminator(sn=True, norm="none", patch=True).eval()
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = D(x)
        assert out.shape[0] == 2 and out.shape[1] == 1

    def test_no_nan(self, discriminator):
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = discriminator(x)
        assert torch.isfinite(out).all()

    def test_backward(self):
        D = Discriminator(patch=False)
        x = torch.randn(2, 3, 512, 512)
        D(x).mean().backward()
        assert any(p.grad is not None for p in D.parameters())


# pipeline

class TestGANPipeline:

    @pytest.mark.parametrize("G_cls", [BaselineGenerator, AdvancedGenerator])
    def test_g_to_d(self, G_cls, discriminator):
        G = G_cls().eval()
        z = torch.randn(2, 128)
        with torch.no_grad():
            fake = G(z)
            score = discriminator(fake)
        assert fake.shape  == (2, 3, 512, 512)
        assert score.shape == (2, 1)


#  dataloader

class TestDataloader:

    def test_batch_shape(self, dataloader):
        dl, cfg = dataloader
        batch = next(iter(dl))
        assert batch.shape == (cfg.batch_size, 3, cfg.input_dim, cfg.input_dim)

    def test_pixel_range(self, dataloader):
        dl, cfg = dataloader
        batch = next(iter(dl))
        assert batch.min() >= -1.0 and batch.max() <= 1.0

    @pytest.mark.parametrize("G_cls", [BaselineGenerator, AdvancedGenerator])
    def test_real_and_fake_through_discriminator(self, dataloader, G_cls):
        dl, cfg = dataloader
        batch = next(iter(dl))

        G = G_cls(n_z=cfg.noise_dim, image_size=cfg.input_dim).eval()
        D = Discriminator(patch=False).eval()

        with torch.no_grad():
            real_score = D(batch)
            fake = G(torch.randn(cfg.batch_size, cfg.noise_dim))
            fake_score = D(fake)

        assert real_score.shape == (cfg.batch_size, 1)
        assert fake_score.shape == (cfg.batch_size, 1)
        assert fake.shape == (cfg.batch_size, 3, cfg.input_dim, cfg.input_dim)
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm



#########################
### GAN Discriminator ###
#########################


class Discriminator(nn.Module):
    """
    Discriminator for GAN frameworks (vanilla GAN, WGAN-GP, HingeGAN).
    """
    def __init__(self, 
                 image_channels: int = 3,
                 base_channels: int = 64,
                 dropout_rate: float = 0.0, 
                 sn: bool = False,
                 neg_slope: float = 0.2,
                 norm: str = "none", 
                 patch: bool = True):
        """
        Args:
            image_channels: Input image channels. Default 3.
            base_channels: Channel width at the first conv block. Default 64.
            dropout_rate: Dropout probability applied in deeper blocks. Default 0.
            sn: Use spectral normalisation on conv layers. Default False.
            neg_slope: LeakyReLU negative slope throughout. Default 0.2.
            norm: Normalisation type: 'batch' | 'instance' | 'none'.
                    - Use 'none' with sn=True (WGAN-GP) or vanilla GAN.
                    - Use 'instance' for WGAN-GP without sn.
                    - Use 'batch' for vanilla / hinge GAN without sn.
            patch:  - If True, use PatchGAN-style output (no global pool).
                    - If False, use global pool + MLP head. Default True.
        """
        super().__init__()

        assert norm in ("batch", "instance", "none"), \
            f"norm must be 'batch', 'instance', or 'none', got '{norm}'"

        self.patch = patch
        ch = base_channels

        self.features = nn.Sequential(
            # 512 -> 256
            self._conv_block(image_channels, ch, sn=sn, norm="none",
                             neg_slope=neg_slope),
            # 256 -> 128
            self._conv_block(ch, ch * 2, sn=sn, norm=norm,
                             neg_slope=neg_slope),
            # 128 -> 64
            self._conv_block(ch * 2, ch * 4, sn=sn, norm=norm,
                             dropout_rate=dropout_rate, neg_slope=neg_slope),
            # 64 -> 32
            self._conv_block(ch * 4, ch * 8, sn=sn, norm=norm,
                             dropout_rate=dropout_rate, neg_slope=neg_slope),
            # 32 -> 16
            self._conv_block(ch * 8, ch * 8, sn=sn, norm=norm,
                             dropout_rate=dropout_rate, neg_slope=neg_slope),
        )

        if patch:
            conv_out = nn.Conv2d(ch * 8, 1, kernel_size=4,
                                 stride=1, padding=1, bias=True)
            self.head = spectral_norm(conv_out) if sn else conv_out
        else:
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(ch * 8 * 4 * 4, ch * 8),
                nn.LeakyReLU(neg_slope),
                nn.Dropout(dropout_rate),
                nn.Linear(ch * 8, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.patch:
            return self.head(x)         # (B, 1, H', W')
        return self.head(self.pool(x))  # (B, 1)

    @staticmethod
    def _conv_block(in_c: int, out_c: int, kernel_size: int = 4,
                    stride: int = 2, padding: int = 1, dropout_rate: float = 0.0,
                    neg_slope: float = 0.2, sn: bool = False, norm: str = "none") -> nn.Sequential:
        conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm == "none"),
        )
        if sn:
            conv = spectral_norm(conv)

        layers: list[nn.Module] = [conv]

        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_c))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_c, affine=True))

        layers.append(nn.LeakyReLU(neg_slope, inplace=True))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))

        return nn.Sequential(*layers)
    


###################
### WGAN Critiс ###
###################



class WGANCritic(nn.Module):
    """Critic for WGAN / WGAN-GP frameworks."""
    def __init__(self,
                 image_channels: int = 3,
                 base_channels: int = 64,
                 dropout_rate: float = 0.0,
                 sn: bool = True,
                 neg_slope: float = 0.2,
                 norm: str = "none",
                 patch: bool = True):
        """
        Args:
            image_channels: Input image channels. Default 3.
            base_channels: Channel width at the first conv block. Default 64.
            dropout_rate: Dropout probability in deeper blocks. Default 0.
            sn: Spectral normalisation on conv layers. Default True.
            neg_slope: LeakyReLU negative slope. Default 0.2.
            norm: Normalisation type: 'instance' | 'layer' | 'none'.
            patch: - If True PatchGAN-style output (B, 1, H', W').
                   - If False -> global pool + MLP head  (B, 1).
        """
        super().__init__()

        assert norm in ("instance", "layer", "none"), (
            f"norm must be 'instance', 'layer', or 'none' for a WGAN critic "
            f"(BatchNorm violates the Lipschitz constraint). Got '{norm}'."
        )

        self.patch = patch
        ch = base_channels

        self.features = nn.Sequential(
            # 512 -> 256
            self._conv_block(image_channels, ch,
                             sn=sn, norm="none", neg_slope=neg_slope),
            # 256 -> 128
            self._conv_block(ch, ch * 2,
                             sn=sn, norm=norm, neg_slope=neg_slope),
            # 128 -> 64
            self._conv_block(ch * 2, ch * 4,
                             sn=sn, norm=norm,
                             dropout_rate=dropout_rate, neg_slope=neg_slope),
            # 64 -> 32
            self._conv_block(ch * 4, ch * 8,
                             sn=sn, norm=norm,
                             dropout_rate=dropout_rate, neg_slope=neg_slope),
            # 32 -> 16
            self._conv_block(ch * 8, ch * 8,
                             sn=sn, norm=norm,
                             dropout_rate=dropout_rate, neg_slope=neg_slope),
        )
        if patch:
            conv_out = nn.Conv2d(ch * 8, 1, kernel_size=4,
                                 stride=1, padding=1, bias=True)
            self.head = spectral_norm(conv_out) if sn else conv_out
        else:
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(ch * 8 * 4 * 4, ch * 8),
                nn.LeakyReLU(neg_slope),
                nn.Dropout(dropout_rate),
                nn.Linear(ch * 8, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.patch:
            return self.head(x)
        return self.head(self.pool(x))

    @staticmethod
    def _conv_block(in_c: int, out_c: int,
                    kernel_size: int = 4, stride: int = 2, padding: int = 1,
                    dropout_rate: float = 0.0, neg_slope: float = 0.2,
                    sn: bool = True, norm: str = "none") -> nn.Sequential:
        conv = nn.Conv2d(in_c, out_c,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=(norm == "none"))
        if sn:
            conv = spectral_norm(conv)

        layers: list[nn.Module] = [conv]

        if norm == "instance":
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        elif norm == "layer":
            layers.append(_ChanLayerNorm(out_c))

        layers.append(nn.LeakyReLU(neg_slope, inplace=True))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))

        return nn.Sequential(*layers)


class _ChanLayerNorm(nn.Module):
    """LayerNorm applied over the channel dimension only (C-first tensors)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> norm over C -> (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
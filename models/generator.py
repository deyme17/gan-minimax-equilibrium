import torch
import torch.nn as nn
import math



###############################################################
### Baseline Generator with Transposed Convolutional blocks ###
###############################################################


class BaselineGenerator(nn.Module):
    """
    Generator for GAN frameworks. Simple vanilla variation.
    """
    def __init__(self,
                 n_z: int = 128,
                 image_size: int = 512,
                 image_channels: int = 3,
                 base_size: int = 8,
                 base_channels: int = 512,
                 ):
        """
        Args:
            n_z: Size of random noise input vector. Default 128.
            image_size: Input spatial resolution (assumed square). Default 512.
            image_channels: Input image channels. Default 3.
            base_size: Feature map width at the first deconv block. Default 8.
            base_channels: Channel width at the first deconv block. Default 512.
        """
        super().__init__()
        self.base_size = base_size
        self.base_channels = base_channels

        self.input_layer = nn.Linear(
            n_z, self.base_size * self.base_size * self.base_channels
        )

        ch = base_channels
        self.upsample = nn.Sequential(
            # 8 -> 16
            self._deconv_block(ch, ch // 2),
            # 16 -> 32
            self._deconv_block(ch // 2, ch // 4),
            # 32 -> 64
            self._deconv_block(ch // 4, ch // 8),
            # 64 -> 128
            self._deconv_block(ch // 8, ch // 8),
            # 128 -> 256
            self._deconv_block(ch // 8, ch // 16),
            # 256 -> 512
            self._deconv_block(ch // 16, image_channels,
                                         last_block=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_layer(z)
        z = z.view(-1, self.base_channels, self.base_size, self.base_size)
        z = self.upsample(z)
        return z

    @staticmethod
    def _deconv_block(in_c: int, out_c: int, kernel_size: int = 4,
                      stride: int = 2, padding: int = 1, last_block: bool = False) -> nn.Sequential:
        deconv = nn.ConvTranspose2d(
            in_c, out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        layers = [deconv]

        if not last_block:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        else:
           layers.append(nn.Tanh())

        return nn.Sequential(*layers)



######################################################################
### Advanced Generator with Upsample-Convolutional Residual blocks ###
######################################################################


class UpsampleResBlock(nn.Module):
    """
    Upsample block with resudual connections.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.activate = nn.ReLU(inplace=True)
        self.identity = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        shortcut = self.identity(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activate(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = (x + shortcut) / math.sqrt(2) # norm var
        x = self.activate(x)

        return x


class AdvancedGenerator(nn.Module):
    """
    Generator for GAN frameworks. Advanced experiment variation.
    """
    def __init__(self,
                 n_z: int = 128,
                 image_size: int = 512,
                 image_channels: int = 3,
                 base_size: int = 8,
                 base_channels: int = 512,
                 ):
        """
        Args:
            n_z: Size of random noise input vector. Default 128.
            image_size: Input spatial resolution (assumed square). Default 512.
            image_channels: Input image channels. Default 3.
            base_size: Feature map width at the first deconv block. Default 8.
            base_channels: Channel width at the first deconv block. Default 512.
        """
        super().__init__()
        self.base_size = base_size
        self.base_channels = base_channels

        self.input_layer = nn.Linear(
            n_z, self.base_size * self.base_size * self.base_channels
        )

        ch = base_channels
        self.upsample_blocks = nn.Sequential(
            # 4 -> 8
            UpsampleResBlock(ch, ch // 2),
            # 8 -> 16
            UpsampleResBlock(ch // 2, ch // 4),
            # 16 -> 32
            UpsampleResBlock(ch // 4, ch // 8),
            # 32 -> 64
            UpsampleResBlock(ch // 8, ch // 16),
            # 64 -> 128
            UpsampleResBlock(ch // 16, ch // 16),
            # 128 -> 256
            UpsampleResBlock(ch // 16, ch // 32),
            # 256 -> 512
            UpsampleResBlock(ch // 32, ch // 32),
        )
        self.out_block = nn.Sequential(
            nn.Conv2d(ch // 32, image_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_layer(z)
        z = z.view(-1, self.base_channels, self.base_size, self.base_size)
        z = self.upsample_blocks(z)
        z = self.out_block(z)
        return z
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
                 base_channels: int = 512,
                 min_channels: int = 16):
        """
        Args:
            n_z: Size of random noise input vector. Default 128.
            image_size: Output spatial resolution (must be a power of 2). Default 512.
            image_channels: Output image channels. Default 3.
            base_channels: Channel width at the first deconv block. Default 512.
            min_channels: Channel floor for intermediate blocks. Default 16.
        """
        super().__init__()

        assert image_size > 0 and (image_size & (image_size - 1)) == 0, \
            f"image_size must be a power of 2, got {image_size}"

        self.base_size = 8
        self.base_channels = base_channels
        n_blocks = int(math.log2(image_size // self.base_size))

        self.input_layer = nn.Linear(
            n_z, self.base_size * self.base_size * base_channels
        )

        # build channel schedule for intermediate blocks
        channel_schedule = []
        in_ch = base_channels
        for _ in range(n_blocks - 1):
            out_ch = max(in_ch // 2, min_channels)
            channel_schedule.append((in_ch, out_ch))
            in_ch = out_ch
        channel_schedule.append((in_ch, image_channels))

        blocks = [
            self._deconv_block(in_c, out_c, last_block=(i == len(channel_schedule) - 1))
            for i, (in_c, out_c) in enumerate(channel_schedule)
        ]
        self.upsample = nn.Sequential(*blocks)

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
        self.norm_sc = nn.BatchNorm2d(in_ch)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        shortcut = self.identity(self.norm_sc(x))

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
                 base_channels: int = 512,
                 min_channels: int = 16):
        """
        Args:
            n_z: Size of random noise input vector.
            image_size: Output spatial resolution (must be a power of 2). Default 512.
            image_channels: Output image channels. Default 3.
            base_channels: Channel width at the first upsample block. Default 512.
            min_channels: Channel floor — no block goes below this. Default 16.
        """
        super().__init__()
        assert image_size > 0 and (image_size & (image_size - 1)) == 0, \
            f"image_size must be a power of 2, got {image_size}"
        
        self.base_size = 4
        self.base_channels = base_channels
        n_blocks = int(math.log2(image_size // self.base_size))

        self.input_layer = nn.Linear(
            n_z, self.base_size * self.base_size * base_channels
        )

        # buils upsample blocks
        blocks = []
        in_ch = base_channels
        for _ in range(n_blocks):
            out_ch = max(in_ch // 2, min_channels)
            blocks.append(UpsampleResBlock(in_ch, out_ch))
            in_ch = out_ch

        self.upsample_blocks = nn.Sequential(*blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(in_ch, image_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_layer(z)
        z = z.view(-1, self.base_channels, self.base_size, self.base_size)
        z = self.upsample_blocks(z)
        z = self.out_block(z)
        return z
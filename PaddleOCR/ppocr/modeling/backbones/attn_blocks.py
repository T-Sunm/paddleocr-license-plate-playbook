import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SEBlock(nn.Layer):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Conv2D(channels, mid, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(mid, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mid = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        
        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2D(in_planes, mid, kernel_size=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(mid, in_planes, kernel_size=1, bias_attr=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Axis 1 is channel dimension in NCHW
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x_cat = paddle.concat([avg_out, max_out], axis=1)
        out = self.conv1(x_cat)
        # CRITICAL FIX: Return feature map modulated by attention mask, not the mask itself
        return x * self.sigmoid(out)

class CBAM(nn.Layer):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = self.sa(x) # SpatialAttention already returns x * mask
        return x

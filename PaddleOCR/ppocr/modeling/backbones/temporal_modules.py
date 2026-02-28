import paddle
from paddle import nn
import paddle.nn.functional as F

class TimeDistributed(nn.Layer):
    """
    Wrapper to run 2D modules (Spatial Stages) on 5D tensors (B, C, T, H, W).
    Standardizes input to 2D for the wrapped module and reconstructs 5D output.
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.transpose([0, 2, 1, 3, 4])  # [B, T, C, H, W]
        x = paddle.reshape(x, [B * T, C, H, W])
        y = self.layer(x)
        
        # Capture new dimensions after spatial module processing
        shape = y.shape
        C_new, H_new, W_new = shape[1], shape[2], shape[3]
        
        # Reshape back to 5D
        y = paddle.reshape(y, [B, T, C_new, H_new, W_new]).transpose([0, 2, 1, 3, 4])
        return y

class BasicConv3D(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bn=True, act=True):
        super().__init__()
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)
        
        self.conv = nn.Conv3D(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias_attr=not bn)
        self.bn = nn.BatchNorm3D(out_ch) if bn else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock3D_Paddle(nn.Layer):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1):
        super().__init__()
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, stride=stride)
        self.conv2 = BasicConv3D(itm_ch, itm_ch, 3)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, act=False)
        
        self.ds = None
        if stride != 1 or in_ch != out_ch:
            self.ds = BasicConv3D(in_ch, out_ch, 1, stride=stride, act=False)

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        return F.relu(y + res)

class TBlockI_Paddle(nn.Layer):
    def __init__(self, in_ch=3, out_ch=48):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv3D(in_ch, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            BasicConv3D(32, out_ch, kernel_size=(3, 3, 3), stride=(1, 2, 2))
        )

    def forward(self, x):
        return self.stem(x)

class TAM_Paddle(nn.Layer):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        self.out_channels = 2 * in_ch
        self.project = None
        if out_ch is not None:
            self.project = nn.Sequential(
                nn.Conv2D(2 * in_ch, out_ch, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_ch),
                nn.ReLU()
            )
            self.out_channels = out_ch

    def forward(self, x):
        avg_pool = paddle.mean(x, axis=2)
        max_pool = paddle.max(x, axis=2)
        out = paddle.concat([avg_pool, max_pool], axis=1)
        if self.project is not None:
            out = self.project(out)
        return out

class NeckAdapter_Paddle(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2D(in_ch, out_ch, kernel_size=3, stride=(2, 1), padding=1, bias_attr=False),
            nn.BatchNorm2D(out_ch),
            nn.ReLU(),
            nn.Conv2D(out_ch, out_ch, kernel_size=3, stride=(2, 1), padding=1, bias_attr=False),
            nn.BatchNorm2D(out_ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, None))
        )

    def forward(self, x):
        return self.adapter(x)

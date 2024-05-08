import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class SauvolaMultiWindow(Module):
    """
    MultiWindow Sauvola Torch Module

    1. Instead of doing Sauvola threshold computation for one window size,
       we do this computation for a list of window sizes.
    2. To speed up the computation over large window sizes,
       we implement the integral feature to compute at O(1).
    3. Sauvola parameters, namely, k and R, can be selected to be
       trainable or not. Detailed meaning of k and R, please refer
       https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
    4. Default R value is made w.r.t. normalized image of range (0, 1)
    """

    def __init__(self,
                 window_size_list=[3, 5, 7, 11, 15, 19],
                 init_k=0.2,
                 init_R=0.5,
                 train_k=True,
                 train_R=True):
        super(SauvolaMultiWindow, self).__init__()
        self.window_size_list = window_size_list
        self.n_wins = len(window_size_list)
        self.init_k = init_k
        self.init_R = init_R
        self.train_k = train_k
        self.train_R = train_R
        self.build()

    def _initialize_ii_buffer(self, x):
        """Compute integeral image
        """
        x_pad = F.pad(x,
                      [0, 0, self.max_wh // 2 + 1, self.max_wh // 2 + 1, self.max_ww // 2 + 1, self.max_ww // 2 + 1, 0,
                       0])
        ii_x = torch.cumsum(x_pad, dim=1)
        ii_x2 = torch.cumsum(ii_x, dim=2)
        return ii_x2

    def _get_max_size(self):
        """Compute the max size of all windows
        """
        mh, mw = 0, 0
        for hw in self.window_size_list:
            if (isinstance(hw, int)):
                h = w = hw
            else:
                h, w = hw[:2]
            mh = max(h, mh)
            mw = max(w, mw)
        return mh, mw

    def build(self):
        self.num_woi = len(self.window_size_list)
        self.count_ii = None
        self.lut = dict()
        self.built = True
        self.max_wh, self.max_ww = self._get_max_size()
        self.k = nn.parameter.Parameter(torch.full(size=[1, self.num_woi, 1, 1, 1], fill_value=self.init_k,
                                                   dtype=torch.float32), requires_grad=self.train_k)

        self.R = nn.parameter.Parameter(torch.full(size=[1, self.num_woi, 1, 1, 1], fill_value=self.init_R,
                                                   dtype=torch.float32), requires_grad=self.train_R)

        return

    def _compute_for_one_size(self, x, x_ii, height, width):
        # 1. compute valid counts for this key
        top = self.max_wh // 2 - height // 2
        bot = top + height
        left = self.max_ww // 2 - width // 2
        right = left + width
        Ay, Ax = (top, left)  # self.max_wh, self.max_ww
        By, Bx = (top, right)  # Ay, Ax + width
        Cy, Cx = (bot, right)  # By + height, Bx
        Dy, Dx = (bot, left)  # Cy, Ax
        ii_key = (height, width)
        top_0 = -self.max_wh // 2 - height // 2 - 1
        bot_0 = top_0 + height
        left_0 = -self.max_ww // 2 - width // 2 - 1
        right_0 = left_0 + width
        Ay0, Ax0 = (top_0, left_0)  # self.max_wh, self.max_ww
        By0, Bx0 = (top_0, right_0)  # Ay, Ax + width
        Cy0, Cx0 = (bot_0, right_0)  # By + height, Bx
        Dy0, Dx0 = (bot_0, left_0)  # Cy, Ax
        # used in testing, where each batch is a sample of different shapes
        counts = torch.ones_like(x[:1, ..., :1])
        count_ii = self._initialize_ii_buffer(counts)
        # compute winsize if necessary
        counts_2d = count_ii[:, Ay:Ay0, Ax:Ax0] \
                    + count_ii[:, Cy:Cy0, Cx:Cx0] \
                    - count_ii[:, By:By0, Bx:Bx0] \
                    - count_ii[:, Dy:Dy0, Dx:Dx0]
        # 2. compute summed feature
        sum_x_2d = x_ii[:, Ay:Ay0, Ax:Ax0] \
                   + x_ii[:, Cy:Cy0, Cx:Cx0] \
                   - x_ii[:, By:By0, Bx:Bx0] \
                   - x_ii[:, Dy:Dy0, Dx:Dx0]
        # 3. compute average feature
        avg_x_2d = sum_x_2d / counts_2d
        return avg_x_2d

    def _compute_for_all_sizes(self, x):
        x_win_avgs = []
        # 1. compute corr(x, window_mean) for different sizes
        # 1.1 compute integral image buffer
        x_ii = self._initialize_ii_buffer(x)

        for hw in self.window_size_list:
            if isinstance(hw, int):
                height = width = hw
            else:
                height, width = hw[:2]
            this_avg = self._compute_for_one_size(x, x_ii, height, width)
            x_win_avgs.append(this_avg)
        return torch.stack(x_win_avgs, dim=1)

    def forward(self, x):
        x = x.double()
        x_2 = x ** 2
        E_x = self._compute_for_all_sizes(x)
        E_x2 = self._compute_for_all_sizes(x_2)
        dev_x = torch.sqrt(torch.maximum(E_x2 - E_x ** 2, torch.tensor(1e-6)))

        T = E_x * (1. + self.k.double() * (dev_x / self.R.double() - 1.))

        return T.float()

    def compute_output_shape(self, input_shape):
        batch_size, n_rows, n_cols, n_chs = input_shape
        return (batch_size, self.num_woi, n_rows, n_cols, n_chs)


class DifferenceThresh(Module):
    def __init__(self,
                 img_min=0.,
                 img_max=1.,
                 init_alpha=16.,
                 train_alpha=False
                 ):
        super().__init__()
        self.img_min = img_min
        self.img_max = img_max
        self.init_alpha = init_alpha
        self.train_alpha = train_alpha
        self.build()

    def build(self):
        self.alpha = nn.parameter.Parameter(torch.full(size=(1, 1, 1, 1), fill_value=self.init_alpha
                                                       , dtype=torch.float32), requires_grad=self.train_alpha)
        return

    def forward(self, inputs):
        img, th = inputs

        scaled_diff = (img - th) * self.alpha / (self.img_max - self.img_min)
        return scaled_diff

    def get_config(self):
        base_config = super().get_config()
        config = {"img_min": self.img_min,
                  "img_max": self.img_max,
                  "init_alpha": self.init_alpha,
                  "train_alpha": self.train_alpha
                  }
        return dict(list(base_config.items()) + list(config.items()))


class InstanceNorm(Module):
    def __init__(self):
        super(InstanceNorm, self).__init__()

    def forward(self, t):
        t_mu = torch.mean(t, dim=(1, 2), keepdim=True)
        t_sigma = torch.maximum(torch.std(t, dim=(1, 2), keepdim=True), torch.tensor(1e-5))
        t_norm = (t - t_mu) / t_sigma
        return t_norm


class Conv_block(Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=(3, 3),
                 dilation_rate=(1, 1),
                 strides=(1, 1),
                 norm_type='inorm',
                 activation='relu',
                 **conv_kwargs,
                 ):
        super(Conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding='same',
                              dilation=dilation_rate, stride=strides)
        # nn.init._no_grad_fill_(self.conv.weight, 0.5)
        # nn.init._no_grad_fill_(self.conv.bias, 0.5)
        if norm_type == 'bnorm':
            self.norm = nn.BatchNorm2d(filters, eps=1e-5, momentum=0, affine=False, track_running_stats=False)
        elif norm_type == 'inorm':
            self.norm = nn.InstanceNorm2d(filters, eps=1e-5, momentum=0, affine=False, track_running_stats=False)
        else:
            raise NotImplementedError(f"ERROR: unknown normalization type {norm_type}")

        self.activation = nn.ReLU()

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        conv = self.conv(x)
        norm = self.norm(conv)
        act = self.activation(norm)
        return act


class SauvolaNet(Module):
    def __init__(self, window_size_list=[3, 5, 7, 11, 15, 19],
                 train_k=True,  # test
                 train_R=True,
                 train_alpha=True,
                 norm_type='inorm',
                 base_filters=4,
                 img_range=(0., 1.)):
        super(SauvolaNet, self).__init__()

        # attention branch
        n = len(window_size_list)
        filters = base_filters
        t = int(np.ceil(np.log2(max(window_size_list)))) - 1
        # 1st block
        self.conv1 = Conv_block(1, filters, norm_type=norm_type)
        # later blocks
        convs = []

        self.convs1 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2), norm_type=norm_type)
        filters += base_filters
        self.convs2 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2), norm_type=norm_type)
        filters += base_filters
        self.convs3 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2), norm_type=norm_type)
        filters += base_filters
        self.convs4 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2), norm_type=norm_type)
        filters += base_filters
        self.convs5 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2), norm_type=norm_type)
        filters += base_filters

        self.conv2 = nn.Conv2d(filters, n, (3, 3), padding='same')
        nn.init.constant_(self.conv2.weight, 0.5)
        nn.init.constant_(self.conv2.bias, 0.5)
        self.softmax = nn.Softmax(dim=1)

        self.th = SauvolaMultiWindow(window_size_list=window_size_list,
                                     train_k=train_k,
                                     train_R=train_R)
        self.diff = DifferenceThresh(img_min=img_range[0],
                                     img_max=img_range[1],
                                     init_alpha=16.,
                                     train_alpha=train_alpha)

    def forward(self, inputs):
        x2 = self.conv1(inputs)
        x = self.convs1(x2)
        x = self.convs2(x)
        x = self.convs3(x)
        x = self.convs4(x)
        x = self.convs5(x)
        f = self.conv2(x)
        f = self.softmax(f)

        x1 = torch.permute(inputs, (0, 2, 3, 1))
        th = self.th(x1)
        th1 = torch.sum(torch.unsqueeze(f, -1) * th, dim=1)
        diff = self.diff([x1, th1])
        diff = torch.permute(diff, (0, 3, 1, 2))
        return diff


if __name__ == '__main__':
    model = SauvolaNet()
    data = torch.ones([3, 1, 256, 256])
    output = model(data)
    print(output.shape)

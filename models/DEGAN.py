import torch
import torch.nn as nn
import torch.nn.functional as F


class DEGAN(nn.Module):
    def __init__(self, biggest_layer=512):
        super(DEGAN, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, biggest_layer // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(biggest_layer // 2, biggest_layer // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(biggest_layer // 2, biggest_layer, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(biggest_layer, biggest_layer, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.drop5 = nn.Dropout(0.5)

        # Decoder
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(biggest_layer, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512 + biggest_layer // 2, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv10 = nn.Conv2d(2, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        d4 = self.drop4(c4)
        p4 = self.pool4(d4)
        c5 = self.conv5(p4)
        d5 = self.drop5(c5)

        # Decoder
        u6 = self.up6(d5)
        merge6 = torch.cat([d4, u6], dim=1)
        c6 = self.conv6(merge6)

        u7 = self.up7(c6)
        merge7 = torch.cat([c3, u7], dim=1)
        c7 = self.conv7(merge7)

        u8 = self.up8(c7)
        merge8 = torch.cat([c2, u8], dim=1)
        c8 = self.conv8(merge8)

        u9 = self.up9(c8)
        merge9 = torch.cat([c1, u9], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        out = self.sigmoid(c10)

        return out


if __name__ == '__main__':
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = DEGAN().cuda()
    res = model(inp)
    print(res.shape)

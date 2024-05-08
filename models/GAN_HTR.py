import torch
import torch.nn as nn


class GAN_HTR(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(GAN_HTR, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.8)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.8)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.8)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.Dropout(0.5)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.Dropout(0.5)
        )

        self.up6 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.8)
        )

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.8)
        )

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.8)
        )

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2, momentum=0.8)
        )

        self.conv10 = nn.Conv2d(2, output_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        merge6 = torch.cat((conv4, up6), dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)
        out = self.sigmoid(conv10)

        return out


if __name__ == '__main__':
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = GAN_HTR().cuda()
    res = model(inp)
    print(res.shape)

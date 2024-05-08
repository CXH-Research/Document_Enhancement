import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, nb_layers=5, nb_filters=64, k_size=3, dropout=0, strides=1):
        super(SAE, self).__init__()

        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.k_size = k_size
        self.dropout = dropout
        self.strides = strides

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.input_img = nn.Conv2d(
            1, nb_filters, kernel_size=k_size, stride=strides, padding=k_size // 2)
        self.bn = nn.BatchNorm2d(nb_filters)
        self.relu = nn.ReLU()
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)

        for i in range(nb_layers):
            self.encoder_layers.append(nn.Conv2d(
                nb_filters, nb_filters, kernel_size=k_size, stride=strides, padding=k_size // 2))
            self.encoder_layers.append(nn.BatchNorm2d(nb_filters))
            self.encoder_layers.append(nn.ReLU())
            if dropout > 0:
                self.encoder_layers.append(nn.Dropout(dropout))

        for i in range(nb_layers):
            self.decoder_layers.append(nn.ConvTranspose2d(
                nb_filters, nb_filters, kernel_size=k_size, stride=strides, padding=k_size // 2))
            self.decoder_layers.append(nn.BatchNorm2d(nb_filters))
            self.decoder_layers.append(nn.ReLU())
            if dropout > 0:
                self.decoder_layers.append(nn.Dropout(dropout))

        self.decoder = nn.Conv2d(
            nb_filters, 1, kernel_size=k_size, stride=1, padding=k_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_img(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout > 0:
            x = self.dropout_layer(x)

        encoder_outputs = []
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_outputs.append(x)

        for i, layer in enumerate(self.decoder_layers):
            ind = self.nb_layers - i - 1
            x = x + encoder_outputs[ind]
            x = layer(x)

        x = self.decoder(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    inp = torch.randn(1, 3, 256, 256).cuda()
    rednet_model = SAE()
    rednet_model = rednet_model.cuda()
    out = rednet_model(inp)
    print(out.shape)

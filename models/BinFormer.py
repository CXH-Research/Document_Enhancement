import math

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
def exists(val):
    return val is not None


def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size=256, patch_size=16, word_size=8, num_classes=1000, dim, depth, heads, mlp_dim=2048,
                 pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0., transformer=None,
                 t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        layers = []
        layer_dim = channels
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)

            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size, stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim, heads=1, depth=1, dim_head=layer_dim, mlp_dim=layer_dim,
                            dropout=dropout) if not is_last else nn.Identity(),
            ])
        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        word_height, word_width = pair(word_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), 
        #    nn.Linear(patch_dim, dim),
        # )
        self.to_patch_embedding_ = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2, dim))

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'cls' else x[:, 0]

        x = self.to_latent(x)
        return 1


class BINMODEL(nn.Module):
    def __init__(
            self,
            *,
            encoder,
            inner_encoder,
            decoder_dim,
            masking_ratio=0.75,
            decoder_depth=1,
            decoder_heads=8,
            decoder_dim_head=64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.InnerEncoder = inner_encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]  # [1,257,768], 257,768
        self.patch_to_emb = encoder.to_patch_embedding
        # self.to_patch - Rearange.. , self.patch_to_emb - Linear Layer
        # pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        pixel_values_per_patch = 256

        # decoder parameters

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        tokens = self.patch_to_emb(img)  # Pass through Linear layer
        b, n, _ = tokens.shape
        tokens = tokens + self.encoder.pos_embedding[:, :(n)]

        encoded_tokens = self.encoder.transformer(tokens)  # Without CLS Token
        # print("Encoded Tokens :",encoded_tokens.shape) # Main Patch

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        # print(decoder_tokens.shape)

        # attend with decoder

        decoded_tokens = self.decoder(decoder_tokens)

        # project to pixel values
        # print("Decoded Tokens Shape",decoded_tokens.shape)
        pred_pixel_values = self.to_pixels(decoded_tokens)

        return pred_pixel_values.unsqueeze(1)


class BinFormer(nn.Module):
    def __init__(
            self,
            *,
            image_size=256
    ):
        super().__init__()
        v = ViT(image_size=image_size, dim=768, depth=6, heads=8)
        IN = ViT(
            image_size=image_size,
            dim=768,
            depth=4,
            heads=6,
            mlp_dim=2048
        )
        self.model = BINMODEL(
            encoder=v,
            inner_encoder=IN,
            masking_ratio=0.5,  # __ doesnt matter for binarization
            decoder_dim=768,
            decoder_depth=6,
            decoder_heads=8  # anywhere from 1 to 8
        )

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':
    image_size = 256
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = BinFormer().cuda()
    res = model(inp)
    print(res.shape)

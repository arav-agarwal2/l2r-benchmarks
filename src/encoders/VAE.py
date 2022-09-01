import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.yamlize import yamlize
from src.constants import DEVICE

from src.encoders.base import BaseEncoder

@yamlize
class VAE(BaseEncoder, torch.nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""
    def __init__(self, image_channels:int=3, image_height:int=384, image_width:int=512, z_dim:int=32, load_from:str=''):
        super().__init__()

        self.im_c = image_channels
        self.im_h = image_height
        self.im_w = image_width
        encoder_list = [
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
        self.encoder = nn.Sequential(*encoder_list)
        sample_img = torch.zeros([1, image_channels, image_height, image_width])
        em_shape = nn.Sequential(*encoder_list[:-1])(sample_img).shape[1:]
        h_dim = np.prod(em_shape)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, em_shape),
            nn.ConvTranspose2d(em_shape[0], 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.load_from = load_from # TODO: Load VAE from saved point, or something.
        # TODO: Figure out where speed encoder should go.

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        #raise ValueError(h.shape)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def encode_raw(self, x: np.ndarray, device) -> np.ndarray:
        # assume x is RGB image with shape (bsz, H, W, 3)
        p = np.zeros([x.shape[0], 42, 144, 3], np.float)
        for i in range(x.shape[0]):
            p[i] = cv2.resize(x[i], (144, 144))[68:110] / 255
        x = p.transpose(0, 3, 1, 2)
        x = torch.as_tensor(x, device=device, dtype=torch.float)
        v = self.representation(x)
        return v, v.detach().cpu().numpy()

    def encode(self, x, device=DEVICE):
        x = torch.as_tensor(x, device=device, dtype=torch.float)
        if len(x.shape) == 3:
            x = x.permute(2,0,1)
            x = torch.unsqueeze(x,0)
            
        else:
            x = x.permute(0,3,1,2)
        h = self.encoder(x)
        #raise ValueError(x.shape,h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z #, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        # expects (N, C, H, W)
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss(self, actual, recon, mu, logvar, kld_weight=1.):
        bce = F.binary_cross_entropy(recon, actual, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return bce + kld * kld_weight

    def update(self, batch_of_images):
        # TODO: Add train method here that makes sense
        pass

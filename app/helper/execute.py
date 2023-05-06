import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pathlib import Path
import calendar
import time

current_GMT = time.gmtime()

time_stamp = calendar.timegm(current_GMT)
BASE_DIR = Path(__file__).resolve(strict=True).parent
IMAGE_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "result"))

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        #
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = nn.Conv2d(3, 64, 7)
        self.in01_1 = InstanceNormalization(64)
        # relu
        self.conv02_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv02_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in02_1 = InstanceNormalization(128)
        # relu
        self.conv03_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv03_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in03_1 = InstanceNormalization(256)
        # relu

        ## res block 1
        self.refpad04_1 = nn.ReflectionPad2d(1)
        self.conv04_1 = nn.Conv2d(256, 256, 3)
        self.in04_1 = InstanceNormalization(256)
        # relu
        self.refpad04_2 = nn.ReflectionPad2d(1)
        self.conv04_2 = nn.Conv2d(256, 256, 3)
        self.in04_2 = InstanceNormalization(256)
        # + input

        ## res block 2
        self.refpad05_1 = nn.ReflectionPad2d(1)
        self.conv05_1 = nn.Conv2d(256, 256, 3)
        self.in05_1 = InstanceNormalization(256)
        # relu
        self.refpad05_2 = nn.ReflectionPad2d(1)
        self.conv05_2 = nn.Conv2d(256, 256, 3)
        self.in05_2 = InstanceNormalization(256)
        # + input

        ## res block 3
        self.refpad06_1 = nn.ReflectionPad2d(1)
        self.conv06_1 = nn.Conv2d(256, 256, 3)
        self.in06_1 = InstanceNormalization(256)
        # relu
        self.refpad06_2 = nn.ReflectionPad2d(1)
        self.conv06_2 = nn.Conv2d(256, 256, 3)
        self.in06_2 = InstanceNormalization(256)
        # + input

        ## res block 4
        self.refpad07_1 = nn.ReflectionPad2d(1)
        self.conv07_1 = nn.Conv2d(256, 256, 3)
        self.in07_1 = InstanceNormalization(256)
        # relu
        self.refpad07_2 = nn.ReflectionPad2d(1)
        self.conv07_2 = nn.Conv2d(256, 256, 3)
        self.in07_2 = InstanceNormalization(256)
        # + input

        ## res block 5
        self.refpad08_1 = nn.ReflectionPad2d(1)
        self.conv08_1 = nn.Conv2d(256, 256, 3)
        self.in08_1 = InstanceNormalization(256)
        # relu
        self.refpad08_2 = nn.ReflectionPad2d(1)
        self.conv08_2 = nn.Conv2d(256, 256, 3)
        self.in08_2 = InstanceNormalization(256)
        # + input

        ## res block 6
        self.refpad09_1 = nn.ReflectionPad2d(1)
        self.conv09_1 = nn.Conv2d(256, 256, 3)
        self.in09_1 = InstanceNormalization(256)
        # relu
        self.refpad09_2 = nn.ReflectionPad2d(1)
        self.conv09_2 = nn.Conv2d(256, 256, 3)
        self.in09_2 = InstanceNormalization(256)
        # + input

        ## res block 7
        self.refpad10_1 = nn.ReflectionPad2d(1)
        self.conv10_1 = nn.Conv2d(256, 256, 3)
        self.in10_1 = InstanceNormalization(256)
        # relu
        self.refpad10_2 = nn.ReflectionPad2d(1)
        self.conv10_2 = nn.Conv2d(256, 256, 3)
        self.in10_2 = InstanceNormalization(256)
        # + input

        ## res block 8
        self.refpad11_1 = nn.ReflectionPad2d(1)
        self.conv11_1 = nn.Conv2d(256, 256, 3)
        self.in11_1 = InstanceNormalization(256)
        # relu
        self.refpad11_2 = nn.ReflectionPad2d(1)
        self.conv11_2 = nn.Conv2d(256, 256, 3)
        self.in11_2 = InstanceNormalization(256)
        # + input

        ##------------------------------------##
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in12_1 = InstanceNormalization(128)
        # relu
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64)
        # relu
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = nn.Conv2d(64, 3, 7)
        # tanh

    def forward(self, x):
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        t04 = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y))))

        ##
        y = F.relu(self.in04_1(self.conv04_1(self.refpad04_1(t04))))
        t05 = self.in04_2(self.conv04_2(self.refpad04_2(y))) + t04

        y = F.relu(self.in05_1(self.conv05_1(self.refpad05_1(t05))))
        t06 = self.in05_2(self.conv05_2(self.refpad05_2(y))) + t05

        y = F.relu(self.in06_1(self.conv06_1(self.refpad06_1(t06))))
        t07 = self.in06_2(self.conv06_2(self.refpad06_2(y))) + t06

        y = F.relu(self.in07_1(self.conv07_1(self.refpad07_1(t07))))
        t08 = self.in07_2(self.conv07_2(self.refpad07_2(y))) + t07

        y = F.relu(self.in08_1(self.conv08_1(self.refpad08_1(t08))))
        t09 = self.in08_2(self.conv08_2(self.refpad08_2(y))) + t08

        y = F.relu(self.in09_1(self.conv09_1(self.refpad09_1(t09))))
        t10 = self.in09_2(self.conv09_2(self.refpad09_2(y))) + t09

        y = F.relu(self.in10_1(self.conv10_1(self.refpad10_1(t10))))
        t11 = self.in10_2(self.conv10_2(self.refpad10_2(y))) + t10

        y = F.relu(self.in11_1(self.conv11_1(self.refpad11_1(t11))))
        y = self.in11_2(self.conv11_2(self.refpad11_2(y))) + t11
        ##

        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(y))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = torch.tanh(self.deconv03_1(self.refpad12_1(y)))

        return y


class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def __call__(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * (
            (n - 1) / float(n)
        )
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out


class ConvLayer(nn.Module):
    """
    conv -> Ins/Batch norm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type='instance'):
        super().__init__()

        # Convolution Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size//2, padding_mode='reflect')

        # Normalization Layers
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

        self.norm_type = norm_type

    def forward(self, x):
        y = self.conv(x)

        if self.norm_type == 'None':
            pass
        else:
            y = self.norm(y)
        return y


class ResidualBlock(nn.Module):
    """
    ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        y = y + residual
        y = F.relu(y, inplace=True)
        return y


class UpsampleConvLayer(nn.Module):
    """
    UpsampleConvLayer
    Upsamples the input and then does a convolution. This method prevent checkboard artifacts
    thus achieve better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm_type='instance'):
        super().__init__()

        self.upsample = upsample

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, padding_mode='reflect')

        # Normalization Layers
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):

        if self.upsample:
            x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)

        out = self.conv2d(x)

        if self.norm_type == 'None':
            pass
        else:
            out = self.norm(out)

        
        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution block
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),  # (b, 32, h, w)
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),  # (b, 64, h//2, w//2)
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),  # (b, 128, h//4, w//4)
            nn.ReLU()
        )

        # Residual block
        self.ResidualBlock = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Deconvolution block
        self.UpSampleBlock = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, 2),  # (b, 64, h//4, w//4)
            nn.ReLU(),
            UpsampleConvLayer(64, 32, 3, 1, 2),  # (b, 32, h//2, w//2)
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm_type='None')  # (b, 3, h, w)
        )

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.ResidualBlock(y)
        y = self.UpSampleBlock(y)
        return y



load_size = 450


def cartoonize(input_image: Image.Image):

    model = Transformer()
    model.load_state_dict(torch.load(f"{BASE_DIR}/Shinkai_net_G_float.pth"))
    model.eval()
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image
    with torch.no_grad():
        output_image = model(input_image)
        output_image = output_image[0]

        # BGR -> RGB
        output_image = output_image[[2, 1, 0], :, :]

        # deprocess, (0, 1)
        output_image = output_image.data.cpu().float() * 0.5 + 0.5
        # save
        output_image_name = "im_gan_"+str(time_stamp)+".jpg"

        vutils.save_image(
            output_image,
            os.path.join(IMAGE_OUTPUT_DIR,  output_image_name)
        )
        
    return {"response": "ok",
            "output": output_image_name}


def load_image(filename, size=None, scale=None):
    img = filename

    if size is not None:
        img = img.resize((size[0], size[1]), Image.ANTIALIAS)
    if scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def stylize(content_image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load content image
    content_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        # mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255))
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1. / 255)),
        # transforms.Normalize(mean=[-0.185, -0.156, -0.106], std=[1, 1, 1]),
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = Autoencoder().to(device)
        state_dict = torch.load(f"{BASE_DIR}/style_model.pth", map_location=torch.device("cpu"))
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
        output = output_transform(output)
        output_image_name = "im_gan_style"+str(time_stamp)+".jpg"

        vutils.save_image(
            output,
            os.path.join(IMAGE_OUTPUT_DIR,  output_image_name)
        )
        
    return {"response": "ok",
            "output": output_image_name}
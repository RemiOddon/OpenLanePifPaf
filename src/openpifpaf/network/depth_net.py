from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
# from cv2 import imwrite
from . import networks
from .layers import disp_to_depth
# from utils import download_model_if_doesnt_exist

class DepthNet(torch.nn.Module):
    """Class to predict for a single image
        """

    def __init__(self, device):
        super().__init__()

        self.device=device
        model_path = os.path.join(os.path.dirname(__file__), 'models/diffnet_640x192')
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = networks.test_hr_encoder.hrnet18(False)
        self.encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()
        
        self.depth_decoder = networks.HRDepthDecoder(self.encoder.num_ch_enc, range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(device)
        self.depth_decoder.eval()
    
    def forward(self, input_image):
        with torch.no_grad():
            # preprocess
            original_width, original_height = input_image.shape[3], input_image.shape[2]
            input_image = transforms.Resize(size=(self.feed_height, self.feed_width))(input_image) #input_image.Resize((self.feed_width, self.feed_height), )
            # input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]

            #disp_resized = disp
            # just like Featdepth
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp_resized, 0.1, 100)

            return depth

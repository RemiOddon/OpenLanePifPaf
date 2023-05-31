"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from . import decoder, logger, network, show, visualizer, __version__
from .predictor import Predictor

LOG = logging.getLogger(__name__)

DLAV_images = False #DLAV
DLAV_images_path = '/home/oddon/DLAV_images'



def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        usage='%(prog)s [options] images',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True,
                        help='Whether to output an image, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='Whether to output a json file, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    return args


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg

def plot_prediction(lanes1,lanes2=None, filename=None):
    lanes1=np.array(lanes1)
    print(lanes1)
    if len(lanes1.shape)!=3:
        lanes1=lanes1.reshape((1,-1,3))
    # Prepare arrays x, y, z
    fig = plt.figure(figsize=(7,7))

    if lanes2 in (None, []) :
        lanes2=[None for _ in lanes1]

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    for lane1, lane2 in zip(lanes1, lanes2):
        x1 = lane1[:,0]
        y1 = lane1[:,1]
        z1 = lane1[:,2]

        if lane2 is not None:
            x2 = lane2[:,0]
            y2 = lane2[:,1]
            z2 = lane2[:,2]

        ax.plot(x1, y1, z1, label='pred', color='b')
        if lane2 is not None:
            ax.plot(x2, y2, z2, label='true', color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('XYZ')

    ax = fig.add_subplot(2, 2, 2)
    for lane1, lane2 in zip(lanes1, lanes2):
        x1 = lane1[:,0]
        y1 = lane1[:,1]
        z1 = lane1[:,2]

        if lane2 is not None:
            x2 = lane2[:,0]
            y2 = lane2[:,1]
            z2 = lane2[:,2]

        ax.plot(y1, z1, label='pred', color='b')
        if lane2 is not None:
            ax.plot(y2, z2, label='true', color='r')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title('YZ')
    
    ax = fig.add_subplot(2, 2, 3)
    for lane1, lane2 in zip(lanes1, lanes2):
        x1 = lane1[:,0]
        y1 = lane1[:,1]
        z1 = lane1[:,2]

        if lane2 is not None:
            x2 = lane2[:,0]
            y2 = lane2[:,1]
            z2 = lane2[:,2]

        ax.plot(x1, z1, label='pred', color='b')
        if lane2 is not None:
            ax.plot(x2, z2, label='true', color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('XZ')

    ax = fig.add_subplot(2, 2, 4)
    for lane1, lane2 in zip(lanes1, lanes2):
        x1 = lane1[:,0]
        y1 = lane1[:,1]
        z1 = lane1[:,2]

        if lane2 is not None:
            x2 = lane2[:,0]
            y2 = lane2[:,1]
            z2 = lane2[:,2]
            
        ax.plot(x1, y1, label='pred', color='b')
        if lane2 is not None:
            ax.plot(x2, y2, label='true', color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('XY')

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.suptitle('3D lane plots (blue: prediction, red: ground truth)')

    plt.show()
    plt.savefig(filename)


def main():
    args = cli()
    annotation_painter = show.AnnotationPainter()

    predictor = Predictor(
        visualize_image=(args.show or args.image_output is not None),
        visualize_processed_image=args.debug,
    )

    if not DLAV_images:
        for pred, _, meta in predictor.images(args.images):
            # json output
            if args.json_output is not None:
                json_out_name = out_name(
                    args.json_output, meta['file_name'], '.predictions.json')
                LOG.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    json.dump([ann.json_data() for ann in pred], f)

            data=json.load(open('/home/oddon/data-openlane/annotations/openlane_keypoints_validation.json'))
            for im in data['images']:
                if args.images == os.path.splitext(im['file_name'])[0]:
                    break

            im_id=im['id']
            ground_truth_lanes=[]
            for lane in data['annotations']:
                if im_id == lane['image_id']:
                    ground_truth_lanes.append(np.array(lane['keypoints']).reshape((-1,4)))
                    
            plot_prediction(pred, ground_truth_lanes, meta['file_name'])
        
    else:
        for video in os.listdir(DLAV_images_path):
            json_file={'project' : '3D Lane Detection by Group 17',
                       'output' : []}

            for img in os.listdir(os.path.join(DLAV_images_path, video)):
                for pred, _, meta in predictor.images(os.path.join(DLAV_images_path, video, img)):
                    frame_pred={'frame' : int(meta['file_name'].split('_')[-1]) , 'prediction' : [ann.json_data() for ann in pred]}
                    json_file['output'].append()

            with open('./' + str(video) + '.json', 'w') as f:
                    json.dump(json_file, f)



if __name__ == '__main__':
    main()
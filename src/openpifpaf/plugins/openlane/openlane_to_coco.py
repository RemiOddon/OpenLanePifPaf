"""
Convert json files of OpenLane into json file with COCO format

Use this command to convert the Openlane dataset :
python3 -m openpifpaf.plugins.openlane.openlane_to_coco --dir_data=/work/scitas-share/datasets/Vita/civil-459/OpenLane/raw --dir_out=/home/oddon/data-openlane
"""

import os
import time
from shutil import copyfile
import json
import argparse

import numpy as np
from PIL import Image

# Packages for data processing, crowd annotations and histograms
try:
    import matplotlib.pyplot as plt  # pylint: disable=import-error
except ModuleNotFoundError as err:
    if err.name != 'matplotlib':
        raise err
    plt = None
try:
    import cv2  # pylint: disable=import-error
except ModuleNotFoundError as err:
    if err.name != 'cv2':
        raise err
    cv2 = None  # pylint: disable=invalid-name

from .constants import NUMBER_KEYPOINTS, LANE_KEYPOINTS, LANE_SKELETON


def cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='data-openlane/train',
                        help='dataset directory')
    parser.add_argument('--dir_out', default='data-openlane',
                        help='where to save annotations and files')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to only process the first 50 images')
    parser.add_argument('--single_sample', action='store_true',
                        help='Whether to only process the first image')
    args = parser.parse_args()
    return args


class OpenlaneToCoco:

    # Prepare json format

    sample = False
    single_sample = False

    def __init__(self, dir_dataset, dir_out):
        """
        :param dir_dataset: Original dataset directory
        :param dir_out: Processed dataset directory
        """

        assert os.path.isdir(dir_dataset), 'dataset directory not found'
        self.dir_dataset = dir_dataset

        assert os.path.isdir(dir_out), "output directory doesn't exits"
        self.dir_out_im = os.path.join(dir_out, 'images')
        self.dir_out_ann = os.path.join(dir_out, 'annotations')
        dir_out_im_train = os.path.join(self.dir_out_im, 'training')
        dir_out_im_val = os.path.join(self.dir_out_im, 'validation')
        os.makedirs(dir_out_im_train, exist_ok=True)
        os.makedirs(dir_out_im_val, exist_ok=True)
        os.makedirs(self.dir_out_ann, exist_ok=True)

        self.json_file = {}

        self.path_train_im=[os.path.join(self.dir_dataset, 'images_training_0'),
                            os.path.join(self.dir_dataset, 'images_training_1'),
                            os.path.join(self.dir_dataset, 'images_training_2'),
                            os.path.join(self.dir_dataset, 'images_training_3'),
                            os.path.join(self.dir_dataset, 'images_training_4'),
                            os.path.join(self.dir_dataset, 'images_training_5'),
                            os.path.join(self.dir_dataset, 'images_training_6'),
                            os.path.join(self.dir_dataset, 'images_training_7'),
                            os.path.join(self.dir_dataset, 'images_training_8'),
                            os.path.join(self.dir_dataset, 'images_training_9'),
                            os.path.join(self.dir_dataset, 'images_training_10'),
                            os.path.join(self.dir_dataset, 'images_training_11'),
                            os.path.join(self.dir_dataset, 'images_training_12'),
                            os.path.join(self.dir_dataset, 'images_training_13'),
                            os.path.join(self.dir_dataset, 'images_training_14'),
                            os.path.join(self.dir_dataset, 'images_training_15')]
        
        self.path_val_im=[os.path.join(self.dir_dataset, 'images_validation_0'),
                          os.path.join(self.dir_dataset, 'images_validation_1'),
                          os.path.join(self.dir_dataset, 'images_validation_2'),
                          os.path.join(self.dir_dataset, 'images_validation_3')]

        # Load train val split
        self.path_train_ann = os.path.join(self.dir_dataset, 'lane3d_1000', 'training')
        self.path_val_ann = os.path.join(self.dir_dataset, 'lane3d_1000', 'validation')

    def process(self):
        """Parse and process the json dataset into a single json file compatible with coco format"""

        for phase, paths_im, paths_ann in zip(['training', 'validation'], [self.path_train_im, self.path_val_im], [self.path_train_ann, self.path_val_ann]):
            cnt_images = 0
            cnt_instances = 0
            self.initiate_json() # Initiate json file at each phase

            for segment in os.listdir(paths_ann):
                segment_json_files = os.listdir(os.path.join(paths_ann, segment))
                segment_json_files = list(np.random.choice(segment_json_files, int( 0.2*len(segment_json_files) ))) # !!! keep 20% of images because of redondancy

                for json_file in segment_json_files:
                    path_json_file = os.path.join(paths_ann, segment, json_file)
                    #Load json file
                    data_json = json.load(open(path_json_file))

                    # Get image path
                    for path_im_folder in paths_im:
                        path_segment_im = os.path.join(path_im_folder, segment)
                        if os.path.isdir(path_segment_im):
                            break
                    
                    im_name = json_file.split('.')[0]
                    path_im = os.path.join(path_segment_im, im_name+'.jpg')
                    im_id = self._process_image(path_im) # added the image to json file
                    cnt_images += 1

                    for idx, lane in enumerate(data_json['lane_lines']):
                        self._process_annotation(lane['xyz'], lane['visibility'], im_id, idx) # add annotations to json file
                        cnt_instances += 1

                    dst = os.path.join(self.dir_out_im, phase, os.path.split(path_im)[-1])
                    #print(path_im, dst)
                    copyfile(path_im, dst)
                
                    if (cnt_images % 1000) == 0:
                        print(f'Treated {cnt_images} images and moved them to {dst}')

            self.save_json_files(phase)
            print(f'\nPhase:{phase}')
            print(f'JSON files directory:  {self.dir_out_ann}')
            print(f'Saved {cnt_instances} instances over {cnt_images} images')

    def save_json_files(self, phase):
        name = 'openlane_keypoints_'
        if self.sample:
            name = name + 'sample_'
        elif self.single_sample:
            name = name + 'single_sample_'

        path_json = os.path.join(self.dir_out_ann, name + phase + '.json')
        with open(path_json, 'w') as outfile:
            json.dump(self.json_file, outfile)

    def _process_image(self, im_path):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_id = int( im_path.split(os.sep)[-2].split('_')[0].split('-')[1] + os.path.splitext(file_name)[0] )  # Numeric code in the image
        im = Image.open(im_path)
        width, height = im.size
        self.json_file["images"].append({
            'coco_url': "unknown",
            'file_name': file_name,
            'id': im_id,
            'license': 1,
            'date_captured': "unknown",
            'width': width,
            'height': height})
        return im_id

    def _process_annotation(self, all_kps, vis, im_id, idx):
        """Process single instance"""

        kps = self._transform_keypoints(all_kps, vis)

        # Enlarge box
        box_tight = [np.min(kps[0, :]), np.min(kps[1, :]), np.min(kps[2, :]),
                     np.max(kps[0, :]), np.max(kps[1, :]), np.max(kps[2, :])]
        lx, ly, lz = box_tight[3] - box_tight[0], box_tight[4] - box_tight[1], box_tight[5] - box_tight[2]
        x_o = box_tight[0] - 0.1 * lx
        y_o = box_tight[1] - 0.1 * ly
        z_o = box_tight[2] - 0.1 * lz
        x_i = box_tight[0] + 1.1 * lx
        y_i = box_tight[1] + 1.1 * ly
        z_i = box_tight[2] + 1.1 * lz

        box = [int(x_o), int(y_o), int(z_o), int(x_i - x_o), int(y_i - y_o), int(z_i - z_o)]  # (x, y, z, lx, ly, lz)

        lane_id = int(str(im_id) + str(idx))
        self.json_file["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': lane_id,
            'area': box[3] * box[4] * box[5],
            'bbox': box,
            'num_keypoints': NUMBER_KEYPOINTS,
            'keypoints': list(kps.T.reshape((-1,))),
            'segmentation': []})
    
    def initiate_json(self):
        """
        Initiate Json for training and val phase
        """
        self.json_file["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                            time.localtime()),
                                description=("Conversion of OpenLane dataset into MS-COCO"
                                            " format with {NUMBER_KEYPOINTS} keypoints"))

        self.json_file["categories"] = [dict(name='lane',
                                        id=1,
                                        skeleton=LANE_SKELETON,
                                        supercategory='lane',
                                        keypoints=LANE_KEYPOINTS)]
        self.json_file["images"] = []
        self.json_file["annotations"] = []

    def _transform_keypoints(self, kps, vis): # Get down/up sampled version of the keypoints, output has NUMBER_KEYPOINTS keypoints
        kps = np.array(kps)
        kps = kps[:, np.argsort(kps[0, :])]
        dist_inter_kp = np.linalg.norm(kps[:, :-1] - kps[:, 1:], axis=0)
        cumdist_inter_kp = np.hstack([[0], dist_inter_kp.cumsum()])
        dist_incr = cumdist_inter_kp[-1]/(NUMBER_KEYPOINTS+1)
        new_kps=np.zeros((4, NUMBER_KEYPOINTS))

        for i in range(NUMBER_KEYPOINTS):
            dist = (i+1) * dist_incr
            for j in range(len(cumdist_inter_kp)):
                if dist < cumdist_inter_kp[j]:
                    alpha = (dist - cumdist_inter_kp[j-1]) / dist_inter_kp[j-1]
                    new_kps[:3, i] = (1-alpha) * kps[:, j-1] + alpha * kps[:, j]
                    new_kps[3, i] = 1 if vis[j-1]==1 and vis[j]==1 else 0
                    break
        return new_kps



def main():
    args = cli()

    # configure
    OpenlaneToCoco.sample = args.sample
    OpenlaneToCoco.single_sample = args.single_sample

    apollo_coco = OpenlaneToCoco(args.dir_data, args.dir_out)
    apollo_coco.process()


if __name__ == "__main__":
    main()

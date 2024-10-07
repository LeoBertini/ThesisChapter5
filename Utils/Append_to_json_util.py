#This code appends to a json project template created first in TagLab (https://taglab.isti.cnr.it/) using a single X-ray groundtruth to label
# Use this code so that all groundtruths are imported automatically into the TagLab project.

#example: update the dictionary with values corresponding to the X-ray images that are going to be labelled
# map_px_to_mm_factor is the CT voxel size
# width and height according to each image
# # {'rect': [0.0, 0.0, 0.0, 0.0],
#          'map_px_to_mm_factor': '0.115457',
#          'width': 1530,
#          'height': 1339,
#          'annotations': [],
#          'layers': [],
#          'channels': [{
#                           'filename': new_filename,
#                           'type': 'RGB'}],
#          'id': new_id,
#          'name': new_name,
#          'workspace': [],
#          'export_dataset_area': [],
#          'acquisition_date': '2024-06-05',
#          'georef_filename': '',
#          'metadata': {},
#          'grid': None}



import json
import os


jsonfile = '/Users/leonardobertini/Downloads/TagLab_main/Coral_LB_0008_SHELL.json'
with open(jsonfile, 'r+') as file:
    # First we load existing data into a dict.
    file_data = json.load(file)


images = []
for file in os.listdir('/Volumes/LeoSegmtSSD/CoralSHELL/TRAINING_DATA/IMAGES'):
    images.append(os.path.join('/Volumes/LeoSegmtSSD/CoralSHELL/TRAINING_DATA/IMAGES',file))

images.sort()
images[0]


for image in images:
    new_filename = '../../../..'+image
    new_id = os.path.basename(image)
    new_name = os.path.basename(image)

    y = {'rect': [0.0, 0.0, 0.0, 0.0],
         'map_px_to_mm_factor': '0.115457',
         'width': 1530,
         'height': 1339,
         'annotations': [],
         'layers': [],
         'channels': [{
                          'filename': new_filename,
                          'type': 'RGB'}],
         'id': new_id,
         'name': new_name,
         'workspace': [],
         'export_dataset_area': [],
         'acquisition_date': '2024-06-05',
         'georef_filename': '',
         'metadata': {},
         'grid': None}

    file_data['images'].append(y)
# convert back to json.

jsonfile_updated = '/Users/leonardobertini/Downloads/TagLab_main/Coral_LB_0008_SHELL_Complete.json'
with open(jsonfile_updated, 'w') as f:
    json.dump(file_data, f, indent=4)
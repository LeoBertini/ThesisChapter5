#Utility code use to mass rename files and move them to desired directory

import cv2
import os
import shutil

#defining directories
dataset_directory = "E:\CoralReefRockSponge"
root_directory = os.path.join(dataset_directory)
images_directory = os.path.join(root_directory, "TRAINING_DATA\\IMAGES")
masks_directory = os.path.join(root_directory, "TRAINING_DATA\\LABELS")

file_list = []
for item in os.listdir(images_directory):
    if not item.startswith('.') and os.path.isfile(os.path.join(images_directory, item)):
        file_list.append(item)

images_filenames = list(sorted(file_list))
correct_images_filenames = [i for i in images_filenames if cv2.imread(os.path.join(images_directory, i)) is not None]

#TODO find corresponding RAW_images
lookup_raw_img_dirs = [ "D:\\PhD\\CORAL_RECONS_RAW\\LB_0034\\TIFF_HorizontalAxis",
                       "D:\\PhD\\CORAL_RECONS_RAW\\LB_0034\\TIFF_VerticalAxis"]

drop_off_dir = images_directory
for dir in lookup_raw_img_dirs:
    files_in_dir = os.listdir(dir)
    for lookup_img in correct_images_filenames:
        name_fixed = lookup_img.replace('_BH','')  #name of 'raw' file to lookup
        for match_img in files_in_dir:
            if name_fixed == match_img and not os.path.isfile(os.path.join(drop_off_dir, name_fixed)):
                print(f"found corresponding file {match_img}")
                src=os.path.join(dir,name_fixed)
                destination=os.path.join(drop_off_dir,name_fixed)
                shutil.copy(src, destination)

            elif os.path.isfile(os.path.join(drop_off_dir, name_fixed)):
                print(f"file already copied  {match_img}")

            else:
                continue


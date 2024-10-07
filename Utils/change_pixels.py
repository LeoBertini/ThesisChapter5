#util script to change label colours

import cv2
import os
import numpy as np

# defining directories
dataset_directory = '/Volumes/LeoSegmtSSD/CoralSHELL'
root_directory = os.path.join(dataset_directory)
images_directory = os.path.join(root_directory, "TRAINING_DATA/IMAGES")
labels_directory = os.path.join(root_directory, "TRAINING_DATA/LABELS")

for img in os.listdir(labels_directory):

    im = cv2.imread(os.path.join(labels_directory, img))
    entered = False
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            if not np.array_equal(im[i][j], [0, 38, 255]) and not np.array_equal(im[i][j], [0, 0, 0]):  # check if not already black or red
                im[i][j] = [0, 0, 0] #change to black
                entered = True

    if entered:
        #save image
        cv2.imwrite(os.path.join(labels_directory, img), im)
    else:
        continue
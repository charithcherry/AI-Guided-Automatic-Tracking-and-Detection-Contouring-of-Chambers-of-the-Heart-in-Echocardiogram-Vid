from AdjustBoundary import adjustContour
from helper import find_files, read_image, selectFrames, save_images
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import os

image_files = find_files()
frNum = int(input('Enter frame number: '))
selectFrames(image_files, frNum-1, frNum)   # Show the selected frames of all the images
i = 0
crop = 30

while i < len(image_files):
    ctr = None
    name = image_files[i].split(".")[0]
    c_save = name + "_inx.txt"

    if os.path.isfile(c_save):           ctr = np.loadtxt(c_save)
    im = read_image(image_files[i], frNum)
    im = im[crop:-crop,crop:-crop]

    exit_mode = adjustContour(im, ctr, im.shape, crop, c_save)
    save_images(image_files[i], frNum)
    if exit_mode == 'exit':                 break
    if exit_mode == 'previous':             i = i - 1
    if exit_mode == 'next':                 i = i + 1
    if exit_mode == 'done':                 pass
    i += 1



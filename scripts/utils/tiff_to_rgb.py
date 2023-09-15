import os
import shutil

import numpy as np
import rasterio as rs
import matplotlib.pyplot as plt
from rasterio.merge import merge
from pathlib import Path


def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def scale(band):
    return band/ 10000.0

main_path = "/Users/aida/Documents/Modules/Thesis/lake-victoria/data/planet/sawa-data/"
product = "PSScene/files/"
regions = ["fujian-1", "fujian-2", "kaloka", "luanda-kotieno", "nyanchebe", "nyaudenge"]

# list to store files
substring = "udm"
#iterate over files in that directory
for loc in regions:
    imgs_path = Path(main_path + loc + "/" + product)
    print('location:',loc)
    substring = "udm"
    for file in os.listdir(imgs_path):
        if file.endswith('.tif') and (substring not in file):
            img = rs.open(main_path + loc + "/" + product + file)
            # PlanetScope 8-band band order
            cb, b, g1, g, y, r, re, n = img.read(masked=True)
            # # PlanetScope 4-band band order: BGRN
            # b, g, r, n = img.read(masked=True)
            #
            #Normalize the bands
            rn = normalize(r)
            gn = normalize(g)
            bn = normalize(b)
            nn = normalize(n)
            # Create RGB natural color composite
            rgb = np.dstack((rn, gn, bn))
            # Let's see how our color composite looks like
            fig = plt.imshow(rgb)
            # save the image
            print(main_path + loc + "/" + product + loc + '-rgb.png')
            plt.imsave(main_path + loc + "/" + product  + loc + '-rgb.png', rgb)


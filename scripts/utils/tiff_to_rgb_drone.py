import os
from pathlib import Path
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def scale(band):
    return band/ 10000.0

def get_file_name(file_path):
    file_path_components = file_path.split('/')
    file_name_and_extension = file_path_components[-1].rsplit('.', 1)
    return file_name_and_extension[0]

main_path = "/Users/aida/Documents/Modules/Thesis/lake-victoria/data/hamilton/"
product = "drone_imgs/"

data = gdal.Open("/Users/aida/Documents/Modules/Thesis/lake-victoria/data/hamilton/drone_imgs/Naya_transparent_mosaic_group1.tif",gdal.GA_ReadOnly)
bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
im = np.stack(bands, 2)

# attempt 1
rgb = im[:,:,[0,1,2]] #see first included image for output
fig = plt.imshow(rgb)
fig.savefig("/Users/aida/Documents/Modules/Thesis/lake-victoria/data/hamilton/drone_imgs/Naya_transparent_mosaic_group1-rgb.png")
# path = Path(main_path + product)
# for file in os.listdir(path):
#     if file.endswith('.tif'):
#         imgs_path = Path(main_path + product + file)
#         img = gdal.Open(imgs_path, gdal.GA_ReadOnly)
#
#



        # # PlanetScope 4-band band order: BGRN
        # img = rasterio.open(main_path + product + file)
        # # sequence of band indexes
        # print(img.indexes)
        # b, g, r, n = img.read(masked=True)
        # #Normalize the bands
        # rn = normalize(r)
        # gn = normalize(g)
        # bn = normalize(b)
        # nn = normalize(n)
        # # Create RGB natural color composite
        # rgb = np.dstack((rn, gn, bn))
        # # Let's see how our color composite looks like
        # # fig = plt.imshow(rgb)
        # # save the image
        # output_name = get_file_name(main_path + product + file)
        # plt.imsave(main_path + product + output_name + '-rgb.png', rgb)
        #

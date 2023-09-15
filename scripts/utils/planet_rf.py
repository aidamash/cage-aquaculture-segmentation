import os
import geopandas as gpd
import pandas as pd
import numpy as np
import shapefile
import matplotlib.pyplot as plt
from osgeo import gdal, osr, ogr

from skimage import exposure
import rasterio as rs
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.features import rasterize
from rasterio import plot
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

from tqdm import tqdm
import time



def normalize(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

def scale(band):
    return (band * (0.00001) * 2.5)


##setting the paths

main_path = '/Users/aida/Documents/Modules/Thesis/lake-victoria/data/planet/lake-area/TOL-3DEC-2022_psscene_analytic_sr_udm2/'
output_path = '/Users/aida/Documents/Modules/Thesis/lake-victoria/data/planet/lake-area/outputs/'
img_name = 'composite.tif'
udm_name = 'composite_udm2.tif'
aoi = 'tol-explorer-aoi.shp'
cage_labels = 'train-data/tol-cage-labels.shp'
water_labels = 'train-data/tol-water-labels.shp'


# Open our image
satdat = rs.open(main_path + img_name)
# satdat is our open dataset object
print(satdat)

udm = rs.open(main_path + udm_name)
aoi = gpd.read_file(main_path + aoi)


# Get the image's coordinate reference system
print(satdat.crs)
# Minimum bounding box in projected units (meters), in our zone
print(satdat.bounds)


## And provides a sequence of band indexes.  These are one indexing, not zero indexing like Numpy arrays.
print(satdat.indexes)

# Get dimensions, in projected units (using the example GeoTIFF, that's meters)
width_in_projected_units = satdat.bounds.right - satdat.bounds.left
height_in_projected_units = satdat.bounds.top - satdat.bounds.bottom
print("Width: {}, Height: {}".format(width_in_projected_units, height_in_projected_units))
# Number of rows and columns (pixels)
print("Rows: {}, Columns: {}".format(satdat.height, satdat.width))

# This dataset's projection uses meters as projected units.
# What are the dimensions of a single pixel in meters?

xres = width_in_projected_units / satdat.width
yres = height_in_projected_units / satdat.height

print(xres, yres)
print("Are the pixels square: {}".format(xres == yres)) #sanity check
# PlanetScope 4-band band order: BGRNIR
b = satdat.read(1)
g = satdat.read(2)
r = satdat.read(3)
n = satdat.read(4)

#scale the bands
rsc = scale(r)
gsc = scale(g)
bsc = scale(b)
nsc = scale(n)
bands_sc = np.array([rsc,gsc,gsc,nsc])

# # Transpose and rescale the image
img = satdat
image = np.array([img.read(3), img.read(2), img.read(1)]).transpose(1,2,0)
p2, p98 = np.percentile(image, (2,98))
image = exposure.rescale_intensity(image, in_range=(p2, p98)) / 100000
image = image.transpose(2,0,1)
#Plot the resulting image
#fig = plt.figure(figsize=(20,12))
#show(image, transform=img.transform)
#plt.imsave(output_path + "tol-rgb.png", image)
print(img.crs)


"""
Calculating water indicies to separate land and water
"""
# Allow division by zero
np.seterr(divide='ignore', invalid='ignore')

ndwi_ep = es.normalized_diff(gsc, nsc)

# check range NDWI values, excluding NaN
np.min(ndwi_ep), np.nanmax(ndwi_ep)

# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# titles = "PlanetScope - Normalized Difference Water Index (NDWI)"
# # Plot your data
# fig, ax = plt.subplots(figsize=(7, 7))
#
# p = plt.imshow(ndwi_ep, cmap="viridis")
# # # Add title and colorbar to show the index
# plt.title(titles)
# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

#plt.colorbar(p, cax=cax)
# save the image
#plt.imsave(output_path + "composit-ndwi.png", ndwi_ep)

"""
Generate the water and land masks
"""

water_mask = np.zeros_like(ndwi_ep)
land_mask = np.zeros_like(ndwi_ep)

water_mask[ndwi_ep >= -0.25] = 1
land_mask[ndwi_ep < -0.25] = 1

# In the image below, colored areas are those identified as water
# plt.title("top of the lake water mask")
#plt.imshow(water_mask)
#plt.imsave(output_path + "watermasekd_img.jpg", water_mask)

def normalize(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))
# # Normalize the bands
rn = normalize(r)
gn = normalize(g)
bn = normalize(b)
nn = normalize(n)
bands_nor = np.array([bn,gn,rn,nn])
# import numpy as np
from skimage import exposure
rgb = np.dstack((rn, gn, bn))
img_masked= rgb * np.repeat(water_mask[:, :, np.newaxis], 3, axis=2)
img_masked_t = img_masked.transpose(1,2,0)
p2, p98 = np.percentile(img_masked_t, (2,98))
img_msked_scaled = exposure.rescale_intensity(img_masked_t, in_range=(p2, p98))
img_msked_scaled_t = img_msked_scaled.transpose(2,0,1)

fig = plt.figure(figsize=(8,8))
plt.imshow(img_msked_scaled_t)
#plt.imsave(output_path + "img_msked.tiff", img_masked)

#
#
# import numpy as np
# from skimage import exposure
# img_to_mask = np.dstack((r, g, b, n))
# img_masked= img_to_mask * np.repeat(water_mask[:, :, np.newaxis], 4, axis=2)
# img_masked_t = img_masked.transpose(1,2,0)
# p2, p98 = np.percentile(img_masked_t, (2,98))
# img_msked_scaled = exposure.rescale_intensity(img_masked_t, in_range=(p2, p98))
# img_msked_scaled_t = img_msked_scaled.transpose(2,0,1)
# print('image_masked_shape', img_msked_scaled_t.shape)
#
# #fig = plt.figure(figsize=(8,8))
# #plt.imshow(img_msked_scaled_t)
# #plt.imsave(output_path + "img_msked_scaled_t.tiff", img_msked_scaled_t)
#
# # #creating image patches
# # i, j = np.where(water_mask > 0)
# # rgb = np.dstack((r, g, b))
# #
# # ind = np.random.randint(0, len(i))
# # window = 20
# # patch = rgb[i[ind] - window:i[ind] + window, j[ind] - window: j[ind] + window]
# #
# # #plt.imshow(10 * patch)
# #
# # """
# # Training data
# # """
# ### Training Data
# cages = gpd.read_file(main_path + cage_labels)
# cages['class_name'] = 'cage'
# cages.head()
#
# water = gpd.read_file(main_path + water_labels)
# water['class_name'] = 'water'
# water.head()
# # Bring the shapefiles into common cordinate system
# cages_prj = cages.to_crs('EPSG:32736')
# water_prj = water.to_crs('EPSG:32736')
#
# # Merge/Combine multiple shapefiles into one
# training_data = gpd.pd.concat([cages_prj, water_prj])
# training_data.head()
# # Export merged geodataframe into shapefile
# # training_data.to_file(main_path + "train-data/tol_training_data.shp")
# training_data.plot(figsize=(8,8))
#
#
# img = gdal.Open(main_path + img_name)
# nrows = img.RasterXSize
# ncols = img.RasterYSize
# print(nrows, ncols)
#
#
#
# gdf = gpd.read_file(main_path + "train-data/tol_training_data.shp")
# gdf.head()
#
# class_ids = training_data['class'].unique()
# class_names = training_data['class_name'].unique()
# (class_ids, class_names)
#
# df = pd.DataFrame({'label': class_names, 'id': class_ids})
# #df.to_csv(main_path + 'train-data/class_lookup.csv')
#
# gdf['id'] = gdf['class']
# print('gdf with ids', gdf.head())
#
# # split the truth data into training and test data sets and save each to a new shapefile
# gdf_train = gdf.sample(frac=0.7)
# gdf_test = gdf.drop(gdf_train.index)
# print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)
# # gdf_train.to_file(main_path + 'train-data/train.shp')
# # gdf_test.to_file(main_path + 'train-data/test.shp')
#
#

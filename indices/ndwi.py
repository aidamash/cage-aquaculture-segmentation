
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from matplotlib.colors import ListedColormap


location = "usenge-anyanga-uwaria-"
product_type = "ps-orthoTile"
main_path = "/Users/aida/Documents/Modules/Thesis/lake_victoria/data/planet"
file_path = "/usenge-anyanga-uwaria/ps-orthoTile/"
file_name = "BGRN_SR.tif"
xml_file = "6180785_3639019_2022-12-31_2465_BGRN_SR_clip.tif"
img = rasterio.open(main_path + file_path + file_name)

xres = (img.bounds.right - img.bounds.left) / img.width
yres = (img.bounds.top - img.bounds.bottom) / img.height
print(xres, yres)
print("Are the pixels square: {}".format(xres == yres))
print(img.crs)
crs = img.crs
# Convert pixel coordinates to world coordinates.

# Upper left pixel
row_min = 0
col_min = 0

# Lower right pixel.  Rows and columns are zero indexing.
row_max = img.height - 1
col_max = img.width - 1

# Transform coordinates with the dataset's affine transformation.
topleft = img.transform * (row_min, col_min)
botright = img.transform * (row_max, col_max)

print("Top left corner coordinates: {}".format(topleft))
print("Bottom right corner coordinates: {}".format(botright))

# All of the metadata required to create an image of the same dimensions, datatype, format, etc. is stored in
# one location.
print(img.meta)
# sequence of band indexes
print(img.indexes)
# Load the 4 bands into 2d arrays - recall that we previously learned PlanetScope band order is BGRN.
# PlanetScope 4-band band order: BGRN
b, g, r, nir = img.read()



# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
rn = normalize(r)
gn = normalize(g)
bn = normalize(b)
nn = normalize(nir)
# Create RGB natural color composite
rgb = np.dstack((rn, gn, bn))

# Let's see how our color composite looks like
fig = plt.imshow(rgb)
# save the image
#plt.imsave(main_path + file_path + location + product_type + '-rgb.png', rgb)
# #
# #Before you can calculate NDWI, normalize the values in the arrays for each band using the Top of Atmosphere (TOA)
# #reflectance coefficients stored in metadata file you downloaded (the .xml file).
# file_name = xml_file
# xmldoc = minidom.parse(main_path + file_path + file_name)
# nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
#
# # XML parser refers to bands by numbers 1-4
# coeffs = {}
# for node in nodes:
#     bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
#     if bn in ['1', '2', '3', '4']:
#         i = int(bn)
#         value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
#         coeffs[i] = float(value)
# print(coeffs)
#
#
# # Multiply by corresponding coefficients
# band_blue = b * coeffs[1]
# band_green = g * coeffs[2]
# band_red = r * coeffs[3]
# band_nir = nir * coeffs[4]
# np.nanmin(band_green), np.nanmax(band_green)
#
#
#
# # Allow division by zero
# np.seterr(divide='ignore', invalid='ignore')
#
# ndwi = (band_green.astype(float) - band_nir.astype(float)) / (band_green + band_nir)
#
# ndwi_ep = es.normalized_diff(gn, nn)
#
# # check range NDWI values, excluding NaN
# print(np.min(ndwi_ep), np.nanmax(ndwi_ep))
#
# titles = [product_type + "- Normalized Difference Water Index (NDWI)"]
# # Turn off bytescale scaling due to float values for NDWI
# ep.plot_bands(ndwi_ep, cmap="Blues", cols=1, title=titles)
#
#
# # Create classes and apply to NDWI results
# ndwi_class_bins = [-1, 0.33, 1]
# ndwi_planet_class = np.digitize(ndwi_ep, ndwi_class_bins)
#
# # Apply the nodata mask to the newly classified NDWI data
# ndwi_planet_class = np.ma.masked_where(
#     np.ma.getmask(ndwi_ep), ndwi_planet_class
# )
# print(np.unique(ndwi_planet_class))
#
# #
# # Define color map
# nbr_colors = ["azure", "darkblue", "tomato"]
# nbr_cmap = ListedColormap(nbr_colors)
#
# # Define class names
# ndwi_cat_names = [
#     "No water",
#     "Deep water",
#     "Others"
# ]
# #
# # # Get list of classes
# classes = np.unique(ndwi_planet_class)
# classes = classes.tolist()
#
# # The mask returns a value of none in the classes. remove that
# classes = classes[0:3]
#
# fig = plt.imshow(ndwi_planet_class, cmap=nbr_cmap)
# # # Plot your data
# # Plot your data
# fig, ax = plt.subplots(figsize=(7, 7))
# im = ax.imshow(ndwi_planet_class, cmap=nbr_cmap)
#
# ep.draw_legend(im_ax=im, classes=classes, titles=ndwi_cat_names)
# ax.set_title(
#     "Sindo - PlanetScope - Normalized Difference Water Index (NDWI) Classes",
#     fontsize=14,
# )
# ax.set_axis_off()
#
# # Auto adjust subplot to fit figure size
# plt.tight_layout()
# #Auto adjust subplot to fit figure size
# plt.tight_layout()
# #
# plt.imsave(main_path + file_path + location + product_type + '-ndwi-class.png', ndwi_planet_class)
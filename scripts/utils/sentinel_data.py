import matplotlib.pyplot as plt
import numpy as np
import rasterio
import os
from osgeo import gdal
import rioxarray

fp_in='/Users/aida/Downloads/Usenge_Sentinel2/'
fp_out=fp_in

#data_path = '/Sindo/HomaBayCounty_Sindo_July_10_sentinel2l1c_analytic/files/'

fn_blue='2023-01-01-00-00_2023-01-01-23-59_Sentinel-2_L1C_B02_(Raw).tiff'
fn_green='2023-01-01-00-00_2023-01-01-23-59_Sentinel-2_L1C_B03_(Raw).tiff'
fn_red='2023-01-01-00-00_2023-01-01-23-59_Sentinel-2_L1C_B04_(Raw).tiff'

band_02 = rioxarray.open_rasterio(fp_in + fn_blue)
b_rpj = band_02.rio.reproject("EPSG:32736")
B_02_rst = b_rpj.rio.to_raster(fp_out + "band_02_rst.tif")

band_03=rioxarray.open_rasterio(fp_in + fn_green)
b_rpj = band_03.rio.reproject("EPSG:32736")
B_03_rst = b_rpj.rio.to_raster(fp_out +"band_03_rst.tif")

band_04=rioxarray.open_rasterio(fp_in + fn_red)
bt_rpj = band_04.rio.reproject("EPSG:32736")
B_04_rst = b_rpj.rio.to_raster(fp_out +"band_04_rst.tif")

b=rasterio.open(fp_in + 'band_02_rst.tif')
g=rasterio.open(fp_in + 'band_03_rst.tif')
r=rasterio.open(fp_in + 'band_04_rst.tif')

red = r.read(1)
green = g.read(1)
blue = b.read(1)

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))

red_n = normalize(red)
green_n = normalize(green)
blue_n = normalize(blue)

rgb_composite_n= np.dstack((red_n, green_n, blue_n))
# plt.imshow(rgb_composite_n)
rgb_plot=plt.imshow(rgb_composite_n, interpolation='lanczos')
#
# rgb_plot=plt.imshow(rgb_composite_n, interpolation='lanczos')
plt.axis('off')
plt.savefig(fp_out + 'sentinel2_rgb_composite.png',dpi=200,bbox_inches='tight')
plt.close('all')
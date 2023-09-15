import os
import shutil
import glob
import numpy as np
import rasterio as rs
import matplotlib.pyplot as plt
from rasterio.merge import merge
from pathlib import Path
from osgeo import  gdal

def scale(band):
    return band/ 10000.0

main_path = "/Users/aida/Documents/Modules/Thesis/lake-victoria/data/planet/sawa-data/"
product = "PSScene/files/"
substring = "udm"
loc = "sindo_8b/"

files_to_mosaic = glob.glob(main_path + loc + product + "*_AnalyticMS_SR_8b_clip.tif")
raster_files = " ".join(files_to_mosaic)
# def merge_tiffs(raster_files):
#     """Take a list of raster files and merge them together"""
#     for p in raster_files:
#         raster_to_mosiac = []
#         raster = rs.open(p)
#         raster_to_mosiac.append(raster)
#         mosaic, output = merge(raster_to_mosiac)
#         output_meta = raster.meta.copy()
#         output_meta.update(
#             {"driver": "GTiff",
#              "height": mosaic.shape[1],
#              "width": mosaic.shape[2],
#              "transform": output,
#              }
#         )

files_to_mosaic = glob.glob(main_path + loc + product + "*_AnalyticMS_SR_8b_clip.tif")
# print(main_path + loc + product + "*_BGRN_Analytic_clip.tif")
files_string = " ".join(files_to_mosaic)
print(files_string)
#
# # #
merge_command ="gdal_merge.py -o /Users/aida/Documents/Modules/Thesis/lake-victoria/data/planet/sawa-data/sindo_8b/PSScene/files/composite.tif -of gtiff " + files_string
print(os.popen(merge_command).read())


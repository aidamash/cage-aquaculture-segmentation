from PIL import Image
import os
import PIL
import glob

width = 640
height = 640

img_path = "/Users/aida/Documents/Modules/Thesis/lake_victoria/object-detection/images/validation/"
images = [file for file in os.listdir(img_path) if file.endswith(('jpeg', 'png', 'jpg'))]
print(images) 
#
for image in images:
    img = Image.open(img_path + image)
    img_resized = img.resize((width, height))
    print(img_resized.size)
    img_resized.save(img_path + "resized_" + image, optimize=True, quality=40)
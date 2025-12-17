import gdown
import os

# Download entire folder from Google Drive
folder_url = 'https://drive.google.com/drive/folders/1y1Wwj65rAHpNVW2cQMdUe-hwqwgwqVPa'
gdown.download_folder(folder_url, output='model_weights', quiet=False)

print('All weights downloaded!')
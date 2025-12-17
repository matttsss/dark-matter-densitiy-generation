import gdown
import os
import shutil

# Download entire folder from Google Drive
folder_url = 'https://drive.google.com/drive/folders/1y1Wwj65rAHpNVW2cQMdUe-hwqwgwqVPa'
gdown.download_folder(folder_url, output='cache', quiet=False)

# Move downloaded dataset to the appropriate directory
shutil.move('cache/BAHAMAS', 'data')
shutil.copytree('cache/MODEL_WEIGHTS', 'model_weights', dirs_exist_ok=True)
shutil.rmtree('cache/MODEL_WEIGHTS')

print('All weights downloaded!')
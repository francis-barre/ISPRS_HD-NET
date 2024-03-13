# %%
import os
from pathlib import Path
from tqdm import tqdm
import cv2

root_dir = Path(__file__).parents[1]

data_path = root_dir / 'data/Inria/train/label/'
cities = ['austin', 'chicago', 'vienna', 'tyrol', 'kitsap']

tiles = os.listdir(data_path)
tiles.sort()  # sort the tiles to ensure consistent order

# %%
for tile in tqdm(tiles):
    label_path = os.path.join(data_path, tile)
    label = cv2.imread(label_path)
    height, width, _ = label.shape
    assert height == 512 and width == 512, (
        f"Tile {tile} has shape {label.shape}")

# %%

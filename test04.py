import glob
import os

path = r"D:\Code\ML\images\Mywork3\card_database_yolo\mosaic\20-21"

for name in os.listdir(path):
    os.rename(os.path.join(path, name), os.path.join(path, name.split('#')[-1]))
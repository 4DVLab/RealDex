import shutil
import os
from pathlib import Path

old_name = "lion"
new_name = "flower_cake"

root_path =Path("/home/lab4dv/data") 
dir_list = ["bags", "ssd", "sda"]

for dir in dir_list:
    for folder in os.listdir(root_path / Path(dir)):
        if folder in old_name:
            for sub_folder in os.listdir(root_path / Path(dir) / Path(folder)):
                if sub_folder.endswith(".md") :
                    continue
                if sub_folder[:len(folder)] in old_name:
                    suffix = sub_folder.split(folder)[1]
                  
                    new_sub_folder = new_name+suffix
                    print(sub_folder, new_sub_folder)
                    shutil.move(root_path/Path(dir)/Path(folder)/Path(sub_folder),root_path/Path(dir)/Path(folder)/Path(new_sub_folder) )
            shutil.move(root_path/Path(dir)/Path(folder), root_path/Path(dir)/Path(new_name))
            print(root_path/Path(dir)/Path(folder), root_path/Path(dir)/Path(new_name))
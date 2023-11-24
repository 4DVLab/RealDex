import numpy as np
from pathlib import Path
import os
import open3d as o3d


bag_folder_path = "/home/lab4dv/data/sda/yogurt/original/yogurt_1_20231105"

model_folder_path = Path(bag_folder_path) / f"models/"
model_path = None
model_name = None
for file_name in os.listdir(model_folder_path):
    if file_name.endswith(".obj"):
        model_name = file_name
        model_path = model_folder_path / file_name
        break

model_mesh = o3d.io.read_triangle_mesh(str(model_path))


matrix = np.array([
[0.967286, -0.230298 ,-0.106394, 1.382632],
[-0.226561 ,-0.595532 ,-0.770721, 0.354687],
[0.114135, 0.769613 ,-0.628227 ,1.275002],
[0.000000 ,0.000000, 0.000000 ,1.000000]

]
)

cam0_to_world_transform = np.array([
    [
        0.3221832568741056,
        0.57349689219082,
        -0.7531927134787391,
        1.9653506143136346
    ],
    [
        0.9353371981154199,
        -0.315619076454712,
        0.15977773436706177,
        -0.114133351627669
    ],
    [
        -0.14608995451977763,
        -0.7559668731004995,
        -0.6381001582534389,
        1.3065810986484299
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0
    ]
])

model_mesh.transform(matrix)

write_path = model_folder_path / f"transform_{model_name}"
o3d.io.write_triangle_mesh(str(write_path),model_mesh)

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
[3.100197194305567838e-01, 9.370643967066634161e-01 -1.606178383274549371e-01 -3.099639028533149876e-01],
[5.812303246073605711e-01, -3.205001809616225827e-01 -7.479645337583880060e-01 -1.920549464690618491e-01],
[-7.523689808338899221e-01, 1.385277965909823272e-01 -6.440116196549796612e-01 2.343811413125115539e+00],
[0.000000000000000000e+00 ,0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00]


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

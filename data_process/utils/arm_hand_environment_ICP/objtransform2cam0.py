import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

def obj2cam0(obj_transform):
        cam3_2_world_transform = {"cam3_rgb_camera_link": [
        [
            0.3169344561946967,
            0.6140455308430225,
            -0.722842055035734,
            1.970698236535785
        ],
        [
            0.934545862031436,
            -0.33219173344541675,
            0.12756443074168414,
            -0.140473081900577
        ],
        [
            -0.16179178667810232,
            -0.7159586149228463,
            -0.6791367163401397,
            1.2722573761206357
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ]}
        world2cam0_transform_inv = np.linalg.inv(cam3_2_world_transform["cam3_rgb_camera_link"])
        obj2cam0 = np.dot(world2cam0_transform_inv,obj_transform)
        quat = Rotation.from_matrix(obj2cam0[:3,:3]).as_quat()
        quat = quat.tolist()
        quat += obj2cam0[:3,3].tolist()
        quat = np.array(quat)
        return quat #(qx,qy,qz,qw,x,y,z)


if __name__ == "__main__":
        # obj_transform = np.array([
        #     [-0.152401 ,0.957698 ,-0.244109, 1.386118 ],
        #     [-0.882691, -0.242996 ,-0.402256 ,0.105873 ],
        #     [-0.444557, 0.154168, 0.882384 ,0.570952 ],
        #     [0.000000, 0.000000 ,0.000000, 1.000000 ]
        #                         ])
        # obj_transform1 = np.array(
        #         [[ 0.98922395, -0.03975979 , 0.14090826 ,-0.10557167],
        #         [ 0.0064417 ,  0.97330777,  0.22941335 ,-0.19275578],
        #         [-0.14626854 ,-0.22603349 , 0.96307548 , 0.21295566],
        #         [ 0.        ,  0.   ,       0.    ,      1.        ]]
        # )

        # obj_transform = obj_transform1 @ obj_transform

        obj_transform = np.array([
                [0.322357 ,0.421495 ,-0.847601, 1.996666 ],
                [-0.503052, 0.834777 ,0.223799, -0.186816 ],
                [0.801888, 0.354244, 0.481131, 0.648614 ],
                [0.000000 ,0.000000, 0.000000, 1.000000 ]
        ])

        result_transform = obj2cam0(obj_transform)
        print(result_transform)
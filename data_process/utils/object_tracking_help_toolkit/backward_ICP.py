import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import json
from scipy.spatial.transform import Rotation
from generate_multiple_view_ply import gen_pcd_for_annotate

#you have to assign the init pose to the icp, so the bug would be less


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtrack_steps", type=int, default=10)
    parser.add_argument("--bag_path", type=str)
    parser.add_argument("--cam_index", type=int, default=0)
    parser.add_argument("--mesh_model_name", type=str)
    return parser.parse_args()


# you have to also try the point to plane ICP
def ICP_between_two_pcd(mesh_model, env_pcd, point_to_point=True):
    transform = None
    if point_to_point:
        source_pcd = mesh_model.sample_points_uniformly(number_of_points=10000)
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, env_pcd, max_correspondence_distance=0.01,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
        transform = icp_result.transformation
    else:
        model_point_cloud = o3d.geometry.PointCloud()
        model_point_cloud.points = mesh_model.vertices
        mesh_model.compute_vertex_normals()
        model_point_cloud.normals = mesh_model.vertex_normals
        icp_result = o3d.pipelines.registration.registration_icp(
            # 最大的crospondence距离，不能设置太小了，不然找不到对应点
            env_pcd, model_point_cloud, max_correspondence_distance=0.004,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=5000)
        )
        # cuz this is the env to icp the model
        transform = np.linalg.inv(icp_result.transformation)
    return transform

# the final target is to gen the ICP transform matrix in the world frame


def load_backtrack_index(bag_path, n_steps_backward):
    bag_path = Path(bag_path)
    backtrack_file_path = bag_path / f"tracking_result/tracking_index.txt"

    tracking_index = 0
    with open(backtrack_file_path, "r") as index_reader:
        tracking_index = np.loadtxt(index_reader).item()
    new_begin_tracking_index = int(round(tracking_index - n_steps_backward))

    return new_begin_tracking_index


def load_cam0rgb_to_world_transform(bag_path):
    all_transform_file_path = Path(bag_path) / f"global_name_position/0.txt"
    with open(all_transform_file_path, "r") as transform_json_reader:
        all_transforms = json.load(transform_json_reader)
    cam0rgb_to_world_tranform = all_transforms["cam0_rgb_camera_link"]
    cam0rgb_to_world_tranform = np.array(
        cam0rgb_to_world_tranform).reshape((4, 4))
    return cam0rgb_to_world_tranform


def seven_num2matrix(seven_num):
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix


def load_mesh_model_in_bag(bag_path, mesh_model_name):
    bag_path = Path(bag_path)
    mesh_model_file_path = bag_path / f"models/{mesh_model_name}.obj"
    mesh_model = o3d.io.read_triangle_mesh(str(mesh_model_file_path))
    return mesh_model


def load_index_pcd_in_bag(bag_path, pcd_index):
    bag_name = bag_path.split("/")[-1]
    pcd_path = Path(bag_path) / f"merged_pcd_filter/merge_pcd_{pcd_index}.ply"
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    return pcd


# input is a mesh model and a env point cloud
def get_index_tracking_result(bag_path, cam_index, tracking_index):
    bag_name = bag_path.split("/")[-1]
    bag_path = Path(bag_path)
    tracking_result_file_path = bag_path / \
        f"tracking_result/{bag_name}_cam_index_{cam_index}_tracking_result.txt"
    with open(tracking_result_file_path, "r") as tracking_result_file_reader:
        tracking_transforms = np.loadtxt(tracking_result_file_reader)
    tracking_index = round(tracking_index)
    transform_seven_num = np.array(
        tracking_transforms[tracking_index]).flatten()
    tracking_index_matrix = seven_num2matrix(transform_seven_num)
    return tracking_index_matrix


def get_new_gt_transform_with_icp(bag_path, new_begin_tracking_index, cam_index, mesh_model_name):
    cam0rgb_to_world_transform = load_cam0rgb_to_world_transform(bag_path)

    tracking_index_matrix = get_index_tracking_result(
        bag_path, cam_index, new_begin_tracking_index)

    tracking_result_transform_world_frame = cam0rgb_to_world_transform @ tracking_index_matrix
    
    mesh_model = load_mesh_model_in_bag(bag_path, mesh_model_name)

    env_pcd = load_index_pcd_in_bag(bag_path, new_begin_tracking_index)

    mesh_model_transform_to_world_frame_with_tracking_result = mesh_model.transform(
        tracking_result_transform_world_frame)

    icp_transform = ICP_between_two_pcd(
        mesh_model_transform_to_world_frame_with_tracking_result, env_pcd)

    mesh_model_icp_in_cam0_frame_transform = icp_transform@tracking_result_transform_world_frame

    return mesh_model_icp_in_cam0_frame_transform


def write_transform_to_gt_file(bag_path, transform, gt_index):
    bag_path = Path(bag_path)
    file_path = bag_path / f"poses/gt{gt_index}.txt"
    with open(file_path, "w") as transform_writer:
        np.savetxt(transform_writer, transform)

def backward_nsteps(bag_path, n_steps_backward, cam_index, mesh_model_name):

    new_begin_tracking_index = load_backtrack_index(bag_path,n_steps_backward)

    new_begin_tracking_index = round(new_begin_tracking_index)

    new_gt_tranform = get_new_gt_transform_with_icp(
        bag_path, new_begin_tracking_index, cam_index, mesh_model_name)
    print(new_gt_tranform)
    write_transform_to_gt_file(
        bag_path, new_gt_tranform, new_begin_tracking_index)


if __name__ == "__main__":
    args = get_args()
    backward_nsteps(args.bag_path, args.backtrack_steps,
                    args.cam_index, args.mesh_model_name)

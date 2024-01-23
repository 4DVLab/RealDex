import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import time
import math
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])

def read_object_initial_pose():
    pass

def load_joints_from_controller_bag(load_path:str):
    ra_path = load_path +"/test_ra_points.txt"
    rh_wr_path = load_path + "/test_rh_wr_points.txt"
    rh_path = load_path + "/test_rh_points.txt"
    ra_joints = np.loadtxt(ra_path)
    rh_wr_joints = np.loadtxt(rh_wr_path)
    rh_joints = np.loadtxt(rh_path)
    return ra_joints, rh_wr_joints, rh_joints

def load_joints_from_dataset(load_path:str):
    joints = np.loadtxt(load_path)
    return joints

def load_joints_from_motion(hand_load_path:str, arm_load_path:str):
    hand_joints = np.loadtxt(hand_load_path)
    arm_joints = np.loadtxt(arm_load_path)

    joints_list = []
    hand_index = 0
    for arm_index in range(len(arm_joints)):
        joints = list(arm_joints[arm_index]) + list(hand_joints[hand_index][1:])
        if 120 < arm_index and hand_index <len(hand_joints)-1:
            hand_index = hand_index + 1
        joints_list.append(joints)
            
    return joints_list

def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    # print(scene_config.gravity)
    # print(scene_config.default_static_friction)
    # print(scene_config.default_dynamic_friction)
    # print(scene_config.default_restitution)
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=-2, z=1)
    viewer.set_camera_rpy(r=0, p=0, y=-math.pi/4)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.fix_root_link = True
    
    # robot: sapien.KinematicArticulation = loader.load_kinematic("/home/lab4dv/IntelligentHand/sapien_ws/robots/shadow_hand/bimanual_srhand_ur.urdf")
    robot: sapien.Articulation = loader.load("/home/lab4dv/IntelligentHand/sapien_ws/robots/shadow_hand/bimanual_srhand_ur.urdf")
    
    robot.set_root_pose(sapien.Pose([0, 0, 0], [0, 0, 0, 1]))
    
    # for x in robot.get_active_joints():
    #     print(x.name)
    
    # exit()
    

    # set objects
    box_builder = scene.create_actor_builder()
    box_half_size = [0.15, 0.14, 0.07]
    box_builder.add_box_collision(half_size= box_half_size)
    box_builder.add_box_visual(half_size=box_half_size, color=[1, 0, 0])
    box = box_builder.build_static(name="box")
    box.set_pose(sapien.Pose(p=[-1.3, 0, box_half_size[2]]))
    

    mesh_builder = scene.create_actor_builder()
    physical_material: sapien.PhysicalMaterial = scene.create_physical_material(
        restitution=0,
        static_friction=2000,
        dynamic_friction=2000
    )
    scale_value = 0.06
    # scale_value = 0.6
    mesh_builder.add_collision_from_file(filename='/home/lab4dv/IntelligentHand/sapien_ws/objects/11demo.obj', material= physical_material, density=50, scale=[scale_value, scale_value, scale_value])
    mesh_builder.add_visual_from_file(filename='/home/lab4dv/IntelligentHand/sapien_ws/objects/11demo.obj', scale=[scale_value, scale_value, scale_value])
    mesh= mesh_builder.build(name='mesh')
    mesh_q = get_quaternion_from_euler(0, 0, -math.pi/2)
    mesh.set_pose(sapien.Pose(p=[-1.26,-0.01,0.19], q=mesh_q))

    joints = load_joints_from_motion("config/hand_points.txt", "config/test_ra_points.txt")
    #initial pose  
    current_index = 0
    target_qpos = list(joints[current_index][1:])
    current_index = current_index + 1
    robot.set_qpos(target_qpos)
    scene.step()
    scene.update_render()
    viewer.render()


    first = 0
    while not viewer.closed:
        first = first + 1
        if first>0:
            for _ in range(4):  # render every 4 steps
                # if balance_passive_force:
                if True:
                    # qf = robot.compute_passive_force(
                    #     gravity=True, 
                    #     coriolis_and_centrifugal=True, 
                    #     external= False
                    # )
                    # robot.set_qf(qf) 
                    target_qpos = list(joints[current_index][1:])
                    current_index = current_index + 1
                    if (current_index >= len(joints)):
                        current_index = 0      
                    robot.set_qpos(target_qpos)
                    # print(robot.get_qpos())
                    if first>30:
                        print(first)
                        time.sleep(0.01)
                scene.step()
                
            
            
        scene.update_render()
        viewer.render()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-root-link', action='store_true')
    parser.add_argument('--balance-passive-force', action='store_true')
    args = parser.parse_args()

    demo(fix_root_link=args.fix_root_link,
         balance_passive_force=args.balance_passive_force)


if __name__ == '__main__':
    main()

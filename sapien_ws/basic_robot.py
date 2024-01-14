import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import time
import math
 
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

def load_joints(load_path:str):
    ra_path = load_path +"/test_ra_points.txt"
    rh_wr_path = load_path + "/test_rh_wr_points.txt"
    rh_path = load_path + "/test_rh_points.txt"
    ra_joints = np.loadtxt(ra_path)
    rh_wr_joints = np.loadtxt(rh_wr_path)
    rh_joints = np.loadtxt(rh_path)
    return ra_joints, rh_wr_joints, rh_joints


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
    
    robot: sapien.KinematicArticulation = loader.load_kinematic("/home/lab4dv/IntelligentHand/sapien_ws/robots/shadow_hand/bimanual_srhand_ur.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [0, 0, 0, 1]))
    # print(x for x in robot.get_joints())

    ra_joints, rh_wr_joints, rh_joints = load_joints("/home/lab4dv/IntelligentHand/drive_ws/bags")


    # find nearest timestamp to match arm and hand movement
    nearest_index =[]
    rh_prt = 0
    for ra_prt in range(len(ra_joints)):
        while rh_prt < len(rh_joints) - 1 and rh_joints[rh_prt][0] < ra_joints[ra_prt][0]:
            rh_prt =  rh_prt + 1
        if rh_prt == 0 or abs(rh_joints[rh_prt][0] - ra_joints[ra_prt][0]) < abs(rh_joints[rh_prt - 1][0] - ra_joints[ra_prt][0]):
            nearest_index.append(rh_prt)
        else:
            nearest_index.append(rh_prt - 1)
        if (ra_prt > nearest_index[ra_prt]):
            print("find nearest index error:", ra_prt, " ", rh_prt)
            return -1
        

    # Set initial joint positions
    # ra_qpos = [-1.985541484947321023e-01, -1.511765714932389759e+00, 1.744144698609367605e+00, -4.199380877807101786e-01, 1.224338149918524188e+00, -2.853568453307711472e+00]
    # rh_wr_qpos = [-3.518346993224432673e-01, 1.084536541826265771e-01]
    # rh_qpos = [-5.977415392589238707e-02, -1.574840546072039660e-01, 5.011600017179467237e-01, -1.065708169943478190e-03, 3.636982963940606495e-01, -3.098749694647516262e-01, -2.618010946901915825e-01, 3.090739293982310617e-03, 4.258698396761410598e-01, 9.958308787068738399e-02, -1.464164005054023332e-01, 4.562792452727346126e-01, 4.423967552310126131e-04, -1.763802015284804403e-01, -1.599434749169263703e-02, 1.721247601542393590e-01, 1.939343557097971404e-01, 3.628516803201176089e-01, 5.766439566088897850e-01, 1.457189073599620549e-02, 1.597839815720200063e-01, -1.169722944991009872e-01]
   
    # for x in robot.get_active_joints():
    #     print(x.get_name())

    # return 

    # set objects
    # box_builder = scene.create_actor_builder()
    # box_half_size = [0.15, 0.14, 0.07]
    # box_builder.add_box_collision(half_size= box_half_size)
    # box_builder.add_box_visual(half_size=box_half_size, color=[1, 0, 0])
    # box = box_builder.build_static(name="box")
    # box.set_pose(sapien.Pose(p=[-1.3, 0, box_half_size[2]]))
    

    mesh_builder = scene.create_actor_builder()
    physical_material: sapien.PhysicalMaterial = scene.create_physical_material(
        restitution=0,
        static_friction=2000,
        dynamic_friction=2000
    )
    # scale_value = 0.06
    scale_value = 0.6
    mesh_builder.add_collision_from_file(filename='/home/lab4dv/IntelligentHand/sapien_ws/objects/fixed_vhacd.obj', material= physical_material, density=50, scale=[scale_value, scale_value, scale_value])
    mesh_builder.add_visual_from_file(filename='/home/lab4dv/IntelligentHand/sapien_ws/objects/fixed_vhacd.obj', scale=[scale_value, scale_value, scale_value])
    mesh= mesh_builder.build(name='mesh')
    mesh_q = get_quaternion_from_euler(0, 0, -math.pi/2)
    mesh.set_pose(sapien.Pose(p=[-1.26,-0.01,0.19], q=mesh_q))


    

    #initial pose  
    ra_current_index = 250
    rh_joint_mess = rh_joints[nearest_index[ra_current_index]][1:]
    rh_joint = list(rh_joint_mess[0:4]) + list(rh_joint_mess[9:17]) + list(rh_joint_mess[4:9])+ list(rh_joint_mess[-5:])
    target_qpos = list(ra_joints[ra_current_index][1:]) + list(rh_wr_joints[ra_current_index][1:]) + rh_joint
    ra_current_index = ra_current_index + 1
    robot.set_qpos(target_qpos)
    scene.step()
    scene.update_render()
    viewer.render()


    first = 0
    while not viewer.closed:
        first = first + 1
        if first>300:
            for _ in range(4):  # render every 4 steps
                # if balance_passive_force:
                if True:
                    # qf = robot.compute_passive_force(
                    #     gravity=True, 
                    #     coriolis_and_centrifugal=True, 
                    #     external= False
                    # )
                    # robot.set_qf(qf) 
                    rh_joint_mess = rh_joints[nearest_index[ra_current_index]][1:]
                    # rh_joint = list(rh_joint_mess[0:4]) + list(rh_joint_mess[9:17]) + list(rh_joint_mess[4:9])+ list(rh_joint_mess[-5:])
                    target_qpos = list(ra_joints[ra_current_index][1:]) + list(rh_wr_joints[ra_current_index][1:]) + rh_joint
                    ra_current_index = ra_current_index + 1
                    if (ra_current_index >= len(ra_joints)):
                        ra_current_index = 0      
                    robot.set_qpos(target_qpos)
                    # print(robot.get_qpos())
                scene.step()
            

            # time.sleep(0.1)
            
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

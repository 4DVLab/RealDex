import os
from models.shadow_hand_builder import ShadowHandBuilder
import re
import numpy as np
from bisect import bisect_left
from tqdm import tqdm, trange
from utils.global_util import tf_to_mat
from main import get_model_name



def interp_data(seq_data, object_transl_seq, time):
    qpos_seq = seq_data['joint_angle']
    tf_seq = seq_data['global_tf']
    
    time_stamp_list = list(qpos_seq.keys())
    time_stamp_list = sorted(time_stamp_list)
    index = bisect_left(time_stamp_list, time)
    before = time_stamp_list[index-1] if index >= 0 else None
    after = time_stamp_list[index] if index <len(time_stamp_list) else None
    
    assert (before is not None and after is not None)
    
    ratio = (time - before) / (after - before)
    b_qpos, a_qpos = qpos_seq[before], qpos_seq[after]
    b_transl, a_transl = tf_seq[before]['rh_wrist'][:3, -1], tf_seq[after]['rh_wrist'][:3, -1]
    b_obj_tranl, a_obj_transl = object_transl_seq[index-1], object_transl_seq[index]
    
    new_transl = b_transl * ratio + a_transl * (1-ratio)
    new_obj_transl = b_obj_tranl * ratio + a_obj_transl * (1-ratio)

    ret_dict = {}
    for key in b_qpos:
        ret_dict[key] = b_qpos[key] * ratio + a_qpos[key] * (1-ratio)
    
    ret_dict['rh_wrist_transl'] = new_transl
    ret_dict['object_transl'] = new_obj_transl
    
    return ret_dict

def export_joint_angle(base_dir, exp_code, start_id, end_id, seq_dur_time = 30, frequency=2):
    '''
    seq_dur_time: duration time of the output seq (second)
    frequency: num of sample in each second (Hz)
    '''
    model_name = get_model_name(exp_code)
    data_dir = os.path.join(base_dir, model_name, exp_code)
    seq_file = os.path.join(data_dir, "TF/tf_seq.npy")
    seq_data = np.load(seq_file, allow_pickle=True).item()
    
    obj_tracking_path = os.path.join(data_dir, "tracking_result/gt_pose.txt")
    object_poses = np.loadtxt(obj_tracking_path)
    object_poses = np.array([tf_to_mat(tf) for tf in object_poses])
    object_transl_seq = object_poses[:, :3, -1]
    
    
    time_stamp_list = list(seq_data['joint_angle'].keys())
    start_time = time_stamp_list[start_id]
    end_time = time_stamp_list[end_id]
    
    sample_num = seq_dur_time * frequency
    qpos_dict = {}
    for id in trange(sample_num):
        time = start_time + (id / sample_num) * (end_time - start_time)
        
        qpos = interp_data(seq_data, object_transl_seq, time)
        print(time)
        qpos_dict[time] = qpos
        
        
    # for id in trange(start_id, end_id+1):
    #     time = time_stamp_list[id]
    #     ret_dict = seq_data['joint_angle'][time]
    #     obj_transl = object_transl_seq[id]
    #     transl = seq_data['global_tf'][time]['rh_wrist'][:3, -1]
        
    #     ret_dict['rh_wrist_transl'] = transl
    #     ret_dict['object_transl'] = obj_transl
    #     qpos_dict[time] = ret_dict
        
        
    return qpos_dict


if __name__ == '__main__':
    base_dir = "/public/home/v-liuym/data/IntelligentHand_data/"
    exp_code = "elephant_watering_can_2_20240110"
    
    
    
    start, end = 0, 50
    qpos_dict = export_joint_angle(base_dir, exp_code, start_id=start, end_id=end)
    
    out_dir = "/public/home/v-liuym/results/output_for_sim/"
    os.makedirs(os.path.join(out_dir, exp_code), exist_ok=True)
    out_path = os.path.join(out_dir, exp_code, f"{start}_{end}.npy")
    np.save(out_path, qpos_dict)
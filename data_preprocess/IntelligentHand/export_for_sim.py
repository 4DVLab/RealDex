import os
from models.shadow_hand_builder import ShadowHandBuilder
import re
import numpy as np
from bisect import bisect_left
from tqdm import tqdm, trange

def get_model_name(exp_code):

    original = "elephant_watering_can_2_20240110"
    '''
    * (_\d+): Matches an underscore followed by one or more digits.
    * (_\d{8})?: Optionally matches another underscore followed by exactly eight digits.
        The ? makes this entire group optional.
    * $: Asserts that this sequence is at the end of the string.
    '''
    pattern = r"(_\d+(_\d{8})?)$"

    # Remove the matched pattern from the original string
    extracted_string = re.sub(pattern, '', original)
    return extracted_string

def interp_qpos(qpos_seq, time):
    time_stamp_list = list(qpos_seq.keys())
    time_stamp_list = sorted(time_stamp_list)
    index = bisect_left(time_stamp_list, time)
    before = time_stamp_list[index-1] if index >= 0 else None
    after = time_stamp_list[index] if index <len(time_stamp_list) else None
    
    assert (before is not None and after is not None)
    
    ratio = (time - before) / (after - before)
    b_qpos = qpos_seq[before]
    a_qpos = qpos_seq[after]
    new_qpos = {}
    for key in b_qpos:
        new_qpos[key] = b_qpos[key] * ratio + a_qpos[key] * (1-ratio)
    
    
    
    return new_qpos

def export_joint_angle(base_dir, exp_code, start_id, end_id, seq_dur_time = 30, frequency=100):
    '''
    seq_dur_time: duration time of the output seq (second)
    frequency: num of sample in each second (Hz)
    '''
    model_name = get_model_name(exp_code)
    data_dir = os.path.join(base_dir, model_name, exp_code)
    seq_file = os.path.join(data_dir, "TF/tf_seq.npy")
    seq_data = np.load(seq_file, allow_pickle=True).item()
    qpos_seq = seq_data['joint_angle']
    time_stamp_list = list(qpos_seq.keys())
    start_time = time_stamp_list[start_id]
    end_time = time_stamp_list[end_id]
    
    sample_num = seq_dur_time * frequency
    qpos_dict = {}
    for id in trange(sample_num):
        time = start_time + (id / sample_num) * (end_time - start_time)
        qpos = interp_qpos(qpos_seq, time)
        print(time)
        qpos_dict[time] = qpos
        
    return qpos_dict


if __name__ == '__main__':
    base_dir = "/public/home/v-liuym/data/IntelligentHand_data/"
    exp_code = "elephant_watering_can_2_20240110"
    start, end = 0, 30
    qpos_dict = export_joint_angle(base_dir, exp_code, start_id=start, end_id=end)
    
    out_dir = "/public/home/v-liuym/results/output_for_sim/"
    os.makedirs(os.path.join(out_dir, exp_code), exist_ok=True)
    out_path = os.path.join(out_dir, exp_code, f"{start}_{end}.npy")
    np.save(out_path, qpos_dict)
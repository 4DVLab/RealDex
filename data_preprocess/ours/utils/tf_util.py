import numpy as np

tf_link_names = [
                # 'WRJ1', 'WRJ0',
                'rh_ffknuckle', 'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal',
                'rh_mfknuckle', 'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal',
                'rh_rfknuckle', 'rh_rfproximal', 'rh_rfmiddle', 'rh_rfdistal',
                'rh_lfmetacarpal', 'rh_lfknuckle', 'rh_lfproximal', 'rh_lfmiddle', 'rh_lfdistal',
                'rh_thbase', 'rh_thproximal', 'rh_thhub', 'rh_thmiddle', 'rh_thdistal',
]

def global_tf_to_joint_angle(global_tf, kintree):
    pass
import os
import pickle

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg
    
    
def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data


def read_annotation(base_dir, seq_name, file_id, split):
    meta_filename = os.path.join(base_dir, split, seq_name, 'meta', file_id + '.pkl')

    _assert_exist(meta_filename)

    pkl_data = load_pickle_data(meta_filename)

    return pkl_data

from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os

import time
import numpy as np
import torch
import json
import sys
sys.path.append('.')
sys.path.append('..')
from data_tools.utils import get_seq_id
from data_tools.utils import OakInk_Transcoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


class AutoRegDataset(Dataset):
    def __init__(self, cfg, mode):
        super.__init__()
        # self.cfg = cfg
        self.dataset_dir = cfg['dataset_dir']
        self.ds_path = os.path.join(self.dataset_dir, mode)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%mode))
        self.ds_mode = mode
        
        self.load_frame_info()
        
        self.load_sbj_info()
        
        self.load_obj_info()

        self.load_on_ram = False
        if cfg['load_on_ram']:
            self.ds = self[:]
            self.load_on_ram = True
            
    def load_frame_info(self):
        frame_names = np.load(os.path.join(self.dataset_dir,self.ds_mode, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([os.path.join(self.dataset_dir, fname) for fname in frame_names])
        self.seq_id = get_seq_id(self.frame_names)
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        
    def load_sbj_info(self):
        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(self.dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(self.dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()
        
        ## Hand vtemps and betas
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        
    def load_obj_info(self):
        
        self.obj_names = list(self.obj_info.keys())

        self.obj_label = {obj_name:i for i, obj_name in enumerate(self.obj_info)}

        ## bps_torch data

        # bps_fname = os.path.join(self.dataset_dir, 'bps.npz')
        # self.bps = torch.from_numpy(np.load(bps_fname)['basis'])
        
        
        
        
    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch
        
        
    def __getitem__(self, item):
        
        return 
    
    


class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False,
                 oakink_path = None):

        super().__init__()

        self.only_params = only_params

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))

        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.seq_id = get_seq_id(self.frame_names)
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        self.obj_names = list(self.obj_info.keys())

        self.obj_label = {obj_name:i for i, obj_name in enumerate(self.obj_info)}

        ## bps_torch data

        bps_fname = os.path.join(dataset_dir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)
        ## Hand vtemps and betas

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

        if oakink_path is None:
            oakink_path = '/ghome/l5/ymliu/data/OakInk/'
        oakink_meta_file = os.path.join(oakink_path, 'shape', 'metaV2', 'object_id.json')
        with open(oakink_meta_file, 'r') as f:
            self.oakink_meta = json.load(f)

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch={}
        for k in data.files:
            if k=='raw_obj_id':
                obj_id = self.obj_label[np.array_str(data[k])]
                data_torch[k] = torch.tensor(obj_id).int()
            elif k=='is_virtual':
                data_torch[k] = torch.tensor(data[k]).bool()
            elif k=='seq_id':
                pass
            else:
                data_torch[k] = torch.tensor(data[k]).float()
        # data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch
    
    def load_disk(self,idx):

        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []
        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        data_out['frame_id'] = torch.tensor(idx).int()

        obj = self.frame_objs[idx].item()
        obj_id = self.obj_label[obj]
        data_out['obj_id'] = torch.tensor(obj_id).int()

        # sbj_id = self.frame_sbjs[idx].item()
        # data_out['hand_shape'] = torch.tensor(self.sbj_info[self.sbjs[sbj_id]]['rh_betas']).float()
        return data_out

if __name__=='__main__':

    data_path = '/ghome/l5/ymliu/data/GrabNet_OakInk_ds/data/'
    ds = LoadData(data_path, ds_name='val', only_params=False)

    dataloader = data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=10, drop_last=True)
    seq_transcoder = OakInk_Transcoder()

    s = time.time()
    for i in range(320):
        a = ds[i]
    print(time.time() - s)
    print('pass')

    dl = iter(dataloader)

    s = time.time()
    for i in range(10):
        a = next(dl)
        f_id = a['frame_id']
        print(f_id[i], ds.seq_id[i])
        print(seq_transcoder(ds.seq_id[i]))
    print(time.time()-s)
    print('pass')

    # mvs = MeshViewers(shape=[1,1])
    #
    # bps_torch = test_ds.bps_torch
    # choice = np.random.choice(range(test_ds.__len__()), 30, replace=False)
    # for frame in choice:
    #     data = test_ds[frame]
    #     rhand = Mesh(v=data['verts_rhand'].numpy(),f=[])
    #     obj = Mesh(v=data['verts_object'].numpy(), f=[], vc=name_to_rgb['blue'])
    #     bps_p = Mesh(v=bps_torch, f=[], vc=name_to_rgb['red'])
    #     mvs[0][0].set_static_meshes([rhand,obj, bps_p])
    #     time.sleep(.4)
    #
    # print('finished')
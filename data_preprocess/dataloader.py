import os
import numpy as np

def load_data_motion(seq_dir):
    list_input_data = []
    list_output_data = []
    list_seq_order = []
    list_seq_trans = []
    
    offset = 0
    for i in range(len(category)):
        filenames = os.listdir(os.path.join(seq_dir, category[i]))

        input_data = np.loadtxt(os.path.join(seq_dir, category[i], 'Input.txt'))
         
        output_data = np.loadtxt(os.path.join(seq_dir, category[i], 'Output.txt'))
         
        seq_order = np.loadtxt(os.path.join(seq_dir, category[i], 'Sequences.txt'), dtype=np.int32)
        seq_trans = np.loadtxt(os.path.join(seq_dir, category[i], 'ModelTrans.txt')).reshape((-1, 4, 4))

        seq_num = np.max(seq_order, axis=0)[0]
        list_input_data += ([None] * seq_num)
        list_output_data += ([None] * seq_num)
        list_seq_order += ([None] * seq_num)
        list_seq_trans += ([None] * seq_num)
        for k in range(seq_num):
            indices = (seq_order[:, 0] == k+1).squeeze()
            list_input_data[k + offset] = input_data[indices, :]
            list_output_data[k + offset] = output_data[indices, :]
            order = seq_order[indices, 1]
            list_seq_order[k + offset] = np.stack([np.ones(order.shape, dtype=np.int32)*i, order], axis=1)
            list_seq_trans[k + offset] = seq_trans[indices]
        offset += seq_num
    return list_seq_order, list_input_data, list_output_data, list_seq_trans

def load_label_dict(dir):
    label_dict = {}
    motion_label = open(dir)
    count = 0
    for line in motion_label:
        label_dict[line.split()[1]] = count
        count += 1 
    motion_label.close()
    return label_dict

def load_contacts(dir, bone2id):
    list_contacts = []
    offset = 0
    for i in range(len(category)):
        file = open(os.path.join(dir, category[i], 'Contacts.txt'))
        lines = file.readlines()
        contacts = []
        c = []
        for line in lines:
            if line == "\n":
                contacts.append(c)
                c = []
                continue
            bone = line.split()[0]
            c.append([int(bone2id[bone]), float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]) 
        contacts = np.array(contacts)   
        file.close()
        list_contacts.append(contacts)

    return list_contacts


class MotionSeqDataset:
    def __init__(self, seq_dir, label_dict_dir):
        """
        Args:
            seq_path(string): Path to the motion sequence file.
        """
        self.seq2model , self.list_input_data, self.list_gt_data, self.list_seq_trans = load_data_motion(seq_dir)
        self.inputlabel_dict = load_label_dict(os.path.join(label_dict_dir, 'InputLabels.txt'))
        self.gtlabel_dict = load_label_dict(os.path.join(label_dict_dir, 'OutputLabels.txt'))
        self.bone2id = {'Hips':1, 'Chest':2, 'Chest2':3, 'Chest3':4, 'Chest4':5, 'Neck':6, 'Head':7, 
                        'RightCollar':8, 'RightShoulder':9, 'RightElbow':10, 'RightWrist':11, 
                        'LeftCollar':12, 'LeftShoulder':13, 'LeftElbow':14, 'LeftWrist':15, 
                        'RightHip':16, 'RightKnee':17, 'RightAnkle':18, 'RightToe':19, 
                        'LeftHip':20, 'LeftKnee':21, 'LeftAnkle':22, 'LeftToe':23}
        self.joints_num = len(self.bone2id)

        self.input_size = len(self.inputlabel_dict) - 16 + self.joints_num * 3
        self.gt_size = len(self.gtlabel_dict)

        self.list_contact = load_contacts(seq_dir, self.bone2id)
       
    def __len__(self):
        return len(self.list_input_data)

    def __getitem__(self, idx):
        return self.seq2model[idx], self.list_input_data[idx][:, 16:], self.list_gt_data[idx], self.list_seq_trans[idx]

    def getFrameNumber(self, idx):
        return self.list_input_data[idx].shape[0]

    def ExtractJointLocalTrans(self, idx, frame):
        seq = self.list_input_data[idx]
        trans = np.zeros(shape=[len(self.bone2id), 4, 4])
        for b_name, b_id in self.bone2id.items():
            for i in range(4):
                for j in range(4):
                    trans[b_id - 1, i, j] = seq[frame, self.inputlabel_dict['Bone' + str(b_id) + b_name +str(i)+str(j)]]
            # positions[:, b_id - 1, 0] = seq[:, self.label_dict['Bone' + str(b_id) + b_name + 'PositionX']]
            # positions[:, b_id - 1, 1] = seq[:, self.label_dict['Bone' + str(b_id) + b_name + 'PositionY']]
            # positions[:, b_id - 1, 2] = seq[:, self.label_dict['Bone' + str(b_id) + b_name + 'PositionZ']]
        return trans

    def ExtractRootsTransform(self, idx, frame):
        seq = self.list_input_data[idx]
        roots = np.zeros(shape=[4, 4])
        for i in range(4):
            for j in range(4):
                roots[i, j] = seq[frame, self.inputlabel_dict['Root'+str(i)+str(j)]]
        return roots

    def ExtractJointGlobalPosition(self, idx, frame):
        local_trans = self.ExtractJointLocalTrans(idx, frame)
        roots = self.ExtractRootsTransform(idx, frame)
        
        position = np.zeros(shape=[len(self.bone2id), 3])
        for b_id in range(len(self.bone2id)):
            trans = np.matmul(roots, local_trans[b_id, :, :])
            position[b_id, :] =  np.matmul(trans, np.array([0, 0, 0, 1]))[:3]

        return position

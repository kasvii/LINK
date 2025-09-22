import sys
sys.path.append('./lib')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp, Block, Attention

from core.config import cfg
from layers.smpl.SMPL import SMPL_layer

import os.path as osp

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
smpl_mean_params = 'data/base_data/smpl_mean_params.npz'

class Pose2Mesh(nn.Module):
    def __init__(self, num_joint, embed_dim=256, SMPL_MEAN_vertices=osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')):
        super(Pose2Mesh, self).__init__()
        
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        
        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            # 1        2       3         4        5          6             7            8          9          10
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            # 11         12        13       14        15        16         17         18       19
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')

        
        self.smpl_joints_name = (
        #  0         1        2        3        4         5        6          7          8         9        10    
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
        # 11       12        13          14        15         16            17          18         19         20  
        'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        #   21        22        23
        'R_Wrist', 'L_Hand', 'R_Hand') # , 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
        self.ances = {'Chest': 'Torso', 'Spine': 'Torso', 'L_Toe': 'L_Ankle', 'R_Toe': 'R_Ankle', 'L_Thorax': 'L_Shoulder', 'R_Thorax': 'R_Shoulder', 'L_Hand': 'L_Wrist', 'R_Hand': 'R_Wrist'}
        
        self.coco_nebors = {'Pelvis': ['Pelvis', 'Neck'], 'L_Hip': ['L_Hip', 'Pelvis'], 'R_Hip': ['R_Hip', 'Pelvis'], 'Torso': ['Pelvis', 'Neck'], 
                       'L_Knee': ['L_Knee', 'L_Hip'], 'R_Knee': ['R_Knee', 'R_Hip'], 'Spine': ['Pelvis', 'Neck'], 'L_Ankle': ['L_Ankle', 'L_Knee'], 
                       'R_Ankle': ['R_Ankle', 'R_Knee'], 'Chest': ['Neck', 'Pelvis'], 'L_Toe': ['L_Ankle', 'L_Knee'], 'R_Toe': ['R_Ankle', 'R_Knee'], 
                       'Neck': ['Neck', 'Nose'], 'L_Thorax': ['Neck', 'L_Shoulder'], 'R_Thorax': ['Neck', 'R_Shoulder'], 'Head': ['Neck', 'Nose'], 
                       'L_Shoulder': ['L_Shoulder', 'Neck'], 'R_Shoulder': ['R_Shoulder', 'Neck'], 'L_Elbow': ['L_Elbow', 'L_Shoulder'], 'R_Elbow': ['R_Elbow', 'R_Shoulder'], 
                       'L_Wrist': ['L_Wrist', 'L_Elbow'], 'R_Wrist': ['R_Wrist', 'R_Elbow'], 'L_Hand': ['L_Wrist', 'L_Elbow'], 'R_Hand': ['R_Wrist', 'R_Elbow']}
        
        self.human36_nebors = {'Pelvis': ['Pelvis', 'Torso'], 'L_Hip': ['L_Hip', 'Pelvis'], 'R_Hip': ['R_Hip', 'Pelvis'], 'Torso': ['Torso', 'Pelvis'], 
                       'L_Knee': ['L_Knee', 'L_Hip'], 'R_Knee': ['R_Knee', 'R_Hip'], 'Spine': ['Torso', 'Pelvis'], 'L_Ankle': ['L_Ankle', 'L_Knee'], 
                       'R_Ankle': ['R_Ankle', 'R_Knee'], 'Chest': ['Neck', 'Torso'], 'L_Toe': ['L_Ankle', 'L_Knee'], 'R_Toe': ['R_Ankle', 'R_Knee'], 
                       'Neck': ['Neck', 'Nose'], 'L_Thorax': ['Neck', 'L_Shoulder'], 'R_Thorax': ['Neck', 'R_Shoulder'], 'Head': ['Head', 'Nose'], 
                       'L_Shoulder': ['L_Shoulder', 'Neck'], 'R_Shoulder': ['R_Shoulder', 'Neck'], 'L_Elbow': ['L_Elbow', 'L_Shoulder'], 'R_Elbow': ['R_Elbow', 'R_Shoulder'], 
                       'L_Wrist': ['L_Wrist', 'L_Elbow'], 'R_Wrist': ['R_Wrist', 'R_Elbow'], 'L_Hand': ['L_Wrist', 'L_Elbow'], 'R_Hand': ['R_Wrist', 'R_Elbow']}
        
        self.smpl_dtype = torch.float32
        self.root_idx_24 = 0
        
        if cfg.DATASET.input_joint_set == 'coco':
            self.nebors, self.joints_name = self.coco_nebors, self.coco_joints_name
        elif cfg.DATASET.input_joint_set == 'human36':
            self.nebors, self.joints_name = self.human36_nebors, self.human36_joints_name
        
        h36m_jregressor = np.load('./data/Human36M/J_regressor_h36m_correct.npy')
        self.smpl = SMPL_layer(
            './data/base_data/SMPL_NEUTRAL.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype,
            num_joints=24
        )
        
        self.joint_encode = nn.Linear(num_joint * 3, 2048)
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        
        self.decshape = nn.Linear(2048, 10)
        self.deccam = nn.Linear(2048, 3)
        self.decphi = nn.Linear(2048, 23 * 2)  # [cos(phi), sin(phi)]
        self.decleaf = nn.Linear(2048, 5 * 4)  # rot_mat quat
        
        self.joint_regs = nn.ModuleList()
        self.joint_lifts_1 = nn.ModuleList()
        self.joint_lifts_2 = nn.ModuleList()
        for i in range(len(self.smpl_joints_name)):
            lifter1 = nn.Linear(3, 512)
            lifter2 = nn.Linear(3, 512)
            regressor = nn.Linear(2048 + 1024, 3)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)
            self.joint_lifts_1.append(lifter1)
            self.joint_lifts_2.append(lifter2)
        

    def forward(self, joints):
        # [B, 17, 3]
        batch_size = joints.shape[0]
        joint_feat = self.joint_encode(joints.view(batch_size, -1)) # [B, 2048]
        xc = joint_feat
        
        xc = xc.view(xc.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

        pred_shape = self.decshape(xc) + init_shape
        pred_camera = self.deccam(xc) + init_cam
        pred_phi = self.decphi(xc)
        pred_leaf = self.decleaf(xc)
        
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        pred_leaf = pred_leaf.reshape(batch_size, 5, 4)
        
        pose = []
        for smpl_joint, reg, lift1, lift2 in zip(self.smpl_joints_name, self.joint_regs, self.joint_lifts_1, self.joint_lifts_2):
            cur_joint = torch.cat((joint_feat, lift1(joints[:, self.joints_name.index(self.nebors[smpl_joint][0])]), lift2(joints[:, self.joints_name.index(self.nebors[smpl_joint][1])])), dim=-1)
            pose.append(reg(cur_joint).unsqueeze(1))
            
        pred_pose = torch.cat(pose, dim=1)
        pred_pose = pred_pose - pred_pose[:, self.root_idx_24, :].unsqueeze(1)
        
        output = self.smpl.hybrik(
            pose_skeleton=pred_pose.type(self.smpl_dtype), # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            leaf_thetas=pred_leaf.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        return  pred_vertices, pred_pose, pred_camera, pred_theta_mats, pred_shape


def get_model(num_joint, embed_dim):
    model = Pose2Mesh(num_joint, embed_dim)

    return model

def test_net():
    print(cfg.DATASET.input_joint_set )
    if cfg.DATASET.input_joint_set == 'coco':
        joint_num = 19
    else:
        joint_num = 17
    batch_size = 3
    model = get_model(joint_num, 128)
    model = model.cuda()
    model.eval()
    joints = torch.randn(batch_size, joint_num, 3).cuda()
    pred = model(joints)
    print(pred.shape)

if __name__ == '__main__':
    test_net()

import os.path as osp
import numpy as np
import math
import torch
import json
import copy
import transforms3d
import scipy.sparse
import cv2
import logging
from pycocotools.coco import COCO

from core.config import cfg 
from graph_utils import build_coarse_graphs
from noise_utils import synthesize_pose

from smpl import SMPL
from coord_utils import world2cam, cam2pixel, process_bbox, rigid_align, get_bbox
from aug_utils import affine_transform, j2d_processing, augm_params, j3d_processing, flip_2d_joint
from Human36M.noise_stats import error_distribution

from funcs_utils import save_obj, stop
from vis import vis_3d_pose, vis_2d_pose, draw_nodes_nodes
import joblib
from _img_utils import split_into_chunks_pose, split_into_chunks_mesh
import pickle
import time
from eval_utils import compute_error_accel

logger = logging.getLogger(__name__)

# from IPython import embed
'''dataset_video dataset_test_comera4'''
'''motionbert detected 2d pose in train and test'''
'''continuos input'''

class Human36M(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        dataset_name = 'Human36M'
        self.debug = cfg.TRAIN.debug
        self.data_split = mode
        self.img_dir = osp.join(cfg.data_dir, dataset_name, 'images')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'annotations')
        self.subject_genders = {1: 'female', 5: 'female', 6: 'male', 7: 'female', 8: 'male', 9: 'male', 11: 'male'}
        self.protocol = 2
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        self.fitting_thr = 25  # milimeter

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.smpl_face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        # H36M joint set
        self.human36_joint_num = 17
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_error_distribution = self.get_stat()
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.joint_regressor_human36 = self.mesh_model.joint_regressor_h36m

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), #(5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.joint_regressor_coco = self.mesh_model.joint_regressor_coco

        self.input_joint_name = cfg.DATASET.input_joint_set  # 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        self.img_paths, self.img_names, self.img_ids, self.bboxs, self.img_hws, self.joint_imgs, self.joint_cams, self.joint_vises, \
        self.poses, self.shapes, self.cam_idxs, self.joint_smpl_cams, self.joints_cam_h36m, self.joint_img_h36ms = self.load_data()
        if self.data_split == 'test':
            det_2d_data_path = osp.join(cfg.data_dir, dataset_name, 'Human36M_test_cpn_joint_2d.json')
            self.datalist_pose2d_det, self.datalist_pose2d_det_name = self.load_pose2d_det(det_2d_data_path)
            print("Check lengths of annotation and detection output: ", len(self.img_paths), len(self.datalist_pose2d_det))
        elif self.data_split == 'train':
            if self.input_joint_name == 'human36':
                det_2d_data_path = osp.join(cfg.data_dir, dataset_name, 'Human36M_train_cpn_joint_2d.json')
                self.datalist_pose2d_det_train, self.datalist_pose2d_det_name_train = self.load_pose2d_det(det_2d_data_path)
            elif self.input_joint_name == 'coco':
                det_2d_data_path = osp.join(cfg.data_dir, dataset_name, 'annotations')
                self.datalist_pose2d_det_train, self.datalist_pose2d_det_name_train = self.load_pose2d_det(det_2d_data_path)
            print("Check lengths of annotation and detection output: ", len(self.img_paths), len(self.datalist_pose2d_det_train))
        self.seqlen = cfg.DATASET.seqlen
        self.stride = cfg.DATASET.stride if self.data_split == 'train' else cfg.DATASET.seqlen
        if cfg.MODEL.name == 'posenet':
            self.vid_indices = split_into_chunks_pose(self.img_names, self.seqlen, self.stride, is_train=(set=='train'))
        elif cfg.MODEL.name == 'pose2mesh_net':
            self.vid_indices = split_into_chunks_mesh(self.img_names, self.seqlen, self.stride, self.poses, is_train=(set=='train'))
        
        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.mesh_model.face, self.joint_num, self.skeleton, self.flip_pairs, levels=9)

    def load_pose2d_det(self, data_path, skip_list=[]):
        pose2d_det = []
        pose2d_det_name = []
        if self.input_joint_name == 'human36':
            with open(data_path) as f:
                data = json.load(f)
                for img_path, pose2d in data.items():
                    pose2d = np.array(pose2d, dtype=np.float32)
                    if self.data_split == 'test' and int(img_path.split('_')[-2]) != 4:
                        continue
                    # pose_list.append({'img_name': img_path, 'pose2d': pose2d})
                    pose2d_det.append(pose2d)
                    pose2d_det_name.append(img_path)
            # pose_list = sorted(pose_list, key=lambda x: x['img_name'])
            perm = np.argsort(pose2d_det_name)
            pose2d_det, pose2d_det_name = np.array(pose2d_det), np.array(pose2d_det_name)
            pose2d_det, pose2d_det_name = pose2d_det[perm], pose2d_det_name[perm]
            subsampling_ratio = self.get_subsampling_ratio()
            if subsampling_ratio != 1:
                new_pose2d_det = []
                new_pose2d_det_name = []
                num = 0
                for idx, item in enumerate(pose2d_det):
                    img_idx = int(pose2d_det_name[idx][-10:-4]) - 1
                    if img_idx % subsampling_ratio == 0:
                        # new_pose_list.append(item)
                        new_pose2d_det.append(item)
                        new_pose2d_det_name.append(pose2d_det_name[idx])
            else:
                new_pose2d_det = pose2d_det
                new_pose2d_det_name = pose2d_det_name   
            new_pose2d_det, new_pose2d_det_name = np.array(new_pose2d_det), np.array(new_pose2d_det_name)
        elif self.input_joint_name == 'coco':
            subject_list = self.get_subject()
            new_pose2d_det = []
            new_pose2d_det_name = []
            joints = {}
            
            for subject in subject_list:
                with open(osp.join(data_path, 'Human36M_subject' + str(subject) + '_joint_coco_img_noise_neuralannot.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)
                    
            for img_name in self.img_names:
                subject = str(int(img_name.split('_')[1]))
                action_idx = str(int(img_name.split('_')[3]))
                subaction_idx = str(int(img_name.split('_')[5]))
                cam_idx = str(int(img_name.split('_')[7]))
                frame_idx = str(int(img_name[-10:-4]) - 1) # frame_idx = img_idx - 1
                pose2d = joints[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)][str(frame_idx)]
                pose2d = np.array(pose2d, dtype=np.float32)
                new_pose2d_det.append(pose2d)
                new_pose2d_det_name.append(img_name)
                
            perm = np.argsort(new_pose2d_det_name)
            new_pose2d_det, new_pose2d_det_name = np.array(new_pose2d_det), np.array(new_pose2d_det_name)
            new_pose2d_det, new_pose2d_det_name = new_pose2d_det[perm], new_pose2d_det_name[perm]
        return new_pose2d_det, new_pose2d_det_name

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 2 # 1 # 5  # 5
        elif self.data_split == 'test':
            return 2 # 1 # 5 # 50 #
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1, 5, 6, 7, 8, 9]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")
        # if self.data_split == 'train':
        #     subject = [1]
        # elif self.data_split == 'test':
        #     subject = [9]
        # else:
        #     assert 0, print("Unknown subset")

        if self.debug:
            subject = subject[0:1]

        return subject

    def get_stat(self):
        ordered_stats = []
        for joint in self.human36_joints_name:
            item = list(filter(lambda stat: stat['Joint'] == joint, error_distribution))[0]
            ordered_stats.append(item)

        return ordered_stats

    def generate_syn_error(self):
        noise = np.zeros((self.human36_joint_num, 2), dtype=np.float32)
        weight = np.zeros(self.human36_joint_num, dtype=np.float32)
        for i, ed in enumerate(self.human36_error_distribution):
            noise[i, 0] = np.random.normal(loc=ed['mean'][0], scale=ed['std'][0])
            noise[i, 1] = np.random.normal(loc=ed['mean'][1], scale=ed['std'][1])
            weight[i] = ed['weight']

        prob = np.random.uniform(low=0.0, high=1.0, size=self.human36_joint_num)
        weight = (weight > prob)
        noise = noise * weight[:, None]

        return noise

    def load_data(self):
        print('Load annotations of Human36M Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        
        img_paths, image_names, img_ids, bboxs, img_hws = [], [], [], [], []
        joint_imgs, joint_cams, joint_vises, poses, shapes = [], [], [], [], []
        joint_smpl_cams, cam_idxs  = [], []
        joint_cams_h36m, joint_img_h36ms = [], []

        for subject in subject_list:
            db = COCO()
            cameras = {}
            joints = {}
            smpl_params = {}
            joints_smpl = {}
            joints_h36m = {}
            
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            if self.input_joint_name == 'human36':
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)
            elif self.input_joint_name == 'coco':
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_coco_cam_3d_neuralannot.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                    joints_h36m[str(subject)] = json.load(f)
            # joint_smpl coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_smpl_cam_smplify.json'), 'r') as f:
                joints_smpl[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_smpl_param_per2frame.json'), 'r') as f:
                smpl_params[str(subject)] = json.load(f)
            
            db.createIndex()

            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = osp.join(self.img_dir, img['file_name'])
                img_name = img_path.split('/')[-1]

                # check subject and frame_idx
                frame_idx = img['frame_idx'];

                if frame_idx % sampling_ratio != 0:
                    continue

                if img_name[:-12] == 's_11_act_02_subact_02_ca_0': # or img_name[:-12] == 's_09_act_10_subact_02_ca_0' or img_name[:-12] == 's_09_act_13_subact_01_ca_0' or img_name[:-12] == 's_11_act_02_subact_02_ca_0' or img_name[:-12] == 's_11_act_06_subact_02_ca_0':
                    continue

                # check smpl parameter exist
                subject = img['subject'];
                action_idx = img['action_idx'];

                subaction_idx = img['subaction_idx'];
                frame_idx = img['frame_idx'];

                # camera parameter
                cam_idx = img['cam_idx']
                
                if self.data_split == 'test' and cam_idx != 4:
                    continue
                cam_param = cameras[str(subject)][str(cam_idx)]
                R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'],
                                                                                  dtype=np.float32), np.array(
                    cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
                cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

                try:
                    smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)][str(frame_idx)]
                except KeyError:
                    # if image_id % 5 == 1:
                    #     skip_idx.append(image_id)
                    #     skip_img_idx.append(img_path.split('/')[-1])
                    #     continue
                    # else:
                    smpl_param = None

                if smpl_param is not None:
                    # smpl_param['gender'] = 'neutral'  # self.subject_genders[subject] # set corresponding gender
                    pose_param = np.array(smpl_param['pose'], dtype=np.float32)
                    shape_param = np.array(smpl_param['shape'], dtype=np.float32)
                else:
                    pose_param = np.zeros(1, dtype=np.float32)
                    shape_param = np.zeros(1, dtype=np.float32)

                # project world coordinate to cam, image coordinate space
                if self.input_joint_name == 'human36':
                    joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                        dtype=np.float32)
                    joint_cam = world2cam(joint_world, R, t)
                    joint_img = cam2pixel(joint_cam, f, c)
                    joint_vis = np.ones((self.human36_joint_num, 1), dtype=np.float32)
                elif self.input_joint_name =='coco':
                    joint_cam = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)][str(frame_idx)], dtype=np.float32)
                    joint_img = cam2pixel(joint_cam, f, c)
                    # joint_img = align_kps_bbox(joint_img, np.array(ann['bbox']))
                    joint_img[:, 2] = 1
                    joint_vis = np.ones((self.coco_joint_num, 1), dtype=np.float32)
                    
                    joint_world_h36m = np.array(joints_h36m[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                        dtype=np.float32)
                    joint_cam_h36m = world2cam(joint_world_h36m, R, t)
                    joint_img_h36m = cam2pixel(joint_cam_h36m, f, c)
                
                
                joint_smpl_cam = np.array(joints_smpl[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)][str(frame_idx)], dtype=np.float32)

                bbox = process_bbox(np.array(ann['bbox'], dtype=np.float32))
                if bbox is None: continue

                img_paths.append(img_path)
                image_names.append(img_name)
                img_ids.append(image_id)
                joint_vises.append(joint_vis)
                bboxs.append(np.array(bbox, dtype=np.float32))
                img_hws.append(np.array((img['height'], img['width']), dtype=np.int32))
                joint_imgs.append(np.array(joint_img, dtype=np.float32))
                joint_cams.append(np.array(joint_cam, dtype=np.float32))
                poses.append(pose_param)
                shapes.append(shape_param)
                cam_idxs.append(cam_idx)
                joint_smpl_cams.append(joint_smpl_cam)
                if self.input_joint_name == 'coco':
                    joint_cams_h36m.append(np.array(joint_cam_h36m, dtype=np.float32))
                    joint_img_h36ms.append(np.array(joint_img_h36m, dtype=np.float32))

        # perm = np.argsort(image_names)
        img_paths, image_names, img_ids, bboxs, img_hws = np.array(img_paths), np.array(image_names), np.array(img_ids, dtype=np.int32), np.array(bboxs), np.array(img_hws)
        joint_imgs, joint_cams, joint_vises, poses, shapes = np.array(joint_imgs), np.array(joint_cams), np.array(joint_vises), np.array(poses), np.array(shapes)
        cam_idxs, joint_smpl_cams = np.array(cam_idxs), np.array(joint_smpl_cams)
        if self.input_joint_name == 'coco':
            joint_cams_h36m = np.array(joint_cams_h36m)
            joint_img_h36ms = np.array(joint_img_h36ms)
            
            
        return img_paths, image_names, img_ids, bboxs, img_hws, joint_imgs, joint_cams, joint_vises, poses, shapes, cam_idxs, joint_smpl_cams, joint_cams_h36m, joint_img_h36ms


    def get_smpl_coord(self, pose_param, shape_param):
        pose, shape, gender = pose_param, shape_param, 'neutral'
        # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_pose = torch.FloatTensor(pose).view(1, -1)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape)
        h36m_joint_coor = torch.matmul(torch.from_numpy(self.joint_regressor_human36), smpl_mesh_coord)
        
        joint_smpl_coor = torch.matmul(self.joint_regressor_smpl, smpl_mesh_coord)
        
        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)
        h36m_joint_coor = h36m_joint_coor.numpy().astype(np.float32).reshape(-1, 3)
        
        joint_smpl_coor = joint_smpl_coor.numpy().astype(np.float32).reshape(-1, 3)
        
        

        # meter -> milimeter
        smpl_mesh_coord *= 1000; smpl_joint_coord *= 1000; h36m_joint_coor *= 1000
        
        joint_smpl_coor *= 1000
        
        # h36m_joint_coor = h36m_joint_coor - h36m_joint_coor[:1]
        # joint_smpl_coor = joint_smpl_coor - joint_smpl_coor[:1]
        
        # draw_nodes_nodes(h36m_joint_coor, joint_smpl_coor)

        return smpl_mesh_coord, h36m_joint_coor, joint_smpl_coor
    
    def get_smpl_coord_only(self, pose_param, shape_param):
        pose, shape, gender = pose_param, shape_param, 'neutral'
        # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_pose = pose.view(1, -1).detach().cpu()
        smpl_shape = shape.view(1, -1).detach().cpu()

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape)
        h36m_joint_coor = torch.matmul(torch.from_numpy(self.joint_regressor_human36), smpl_mesh_coord)
        
        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3)
        h36m_joint_coor = h36m_joint_coor.numpy().astype(np.float32).reshape(-1, 3)

        # meter -> milimeter
        smpl_mesh_coord *= 1000; smpl_joint_coord *= 1000; h36m_joint_coor *= 1000
    
        return smpl_mesh_coord, h36m_joint_coor

    def get_fitting_error(self, h36m_joint, smpl_mesh):
        h36m_joint = h36m_joint - h36m_joint[self.human36_root_joint_idx,None,:] # root-relative

        h36m_from_smpl = np.dot(self.joint_regressor_human36, smpl_mesh)
        # translation alignment
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:]
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def get_coco_from_mesh(self, mesh_coord_cam, cam_param):
        # regress coco joints
        joint_coord_cam = np.dot(self.joint_regressor_coco, mesh_coord_cam)
        joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        f, c = cam_param['focal'], cam_param['princpt']
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.coco_joints_name.index('L_Shoulder')
        rshoulder_idx = self.coco_joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def __len__(self):
        # return len(self.datalist)
        return len(self.vid_indices)
    
    def __getitem__(self, idx):
        return self.get_single_item(idx)
    
    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return list(range(start_index, end_index+1))
        else:
            final_data = []
            single_data = data[start_index]
            for i in range(self.seqlen):
                final_data.append(single_data)
            return final_data

    def get_single_item(self, idx):
        start_index, end_index = self.vid_indices[idx]
        
        joint_imgs = []
        joint_cams, joint_cam_h36ms, joint_img_oris, pose_params, shape_params, joint_smpl_coors = [], [], [], [], [], []
        lift_joint_valids, reg_joint_valids = [], []
        flip, rot = augm_params(is_train=(self.data_split == 'train'))
        for num in range(self.seqlen):
            if start_index == end_index:
                single_idx = start_index
            else:
                single_idx = start_index + num
            img_id, bbox, img_shape = self.img_ids[single_idx], self.bboxs[single_idx].copy(), self.img_hws[single_idx]
            
            if len(self.poses[single_idx]) != 1:
                pose_param = self.poses[single_idx].copy()
                shape_param = self.shapes[single_idx].copy()
            else:
                pose_param = None
                shape_param = None
                
            img_name = self.img_names[single_idx]

            # h36m joints from datasets
            if self.input_joint_name == 'human36':
                joint_cam_h36m, joint_img_h36m = self.joint_cams[single_idx].copy(), self.joint_imgs[single_idx].copy()
                root_coor = joint_cam_h36m[:1]
                joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]
                joint_img, joint_cam = joint_img_h36m.copy(), joint_cam_h36m.copy()
            elif self.input_joint_name == 'coco':
                joint_cam_h36m = self.joints_cam_h36m[single_idx].copy()
                root_coord = joint_cam_h36m[:1].copy()
                joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]
                joint_cam_coco, joint_img_coco = self.joint_cams[single_idx].copy(), self.joint_imgs[single_idx].copy()
                joint_cam_coco = joint_cam_coco - root_coord # joint_cam_coco[-2:-1]
                joint_img, joint_cam = joint_img_coco.copy(), joint_cam_coco.copy()
            
            # smpl joints from datasets
            joint_smpl_cam = self.joint_smpl_cams[single_idx].copy()
            joint_smpl_cam = joint_smpl_cam - joint_smpl_cam[self.smpl_root_joint_idx]
            
            # make new bbox
            tight_bbox = get_bbox(joint_img)
            bbox = process_bbox(tight_bbox.copy())
            # aug
            # joint_img, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox, rot, 0, None)
            if cfg.MODEL.name == 'pose2mesh_net':
                # joint_img_ori, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox, rot, 0, None)
                if self.input_joint_name == 'coco':
                    joint_img_ori = self.joint_img_h36ms[single_idx].copy()
                else:
                    joint_img_ori = self.joint_imgs[single_idx].copy()
                if flip:
                    joint_img_ori = flip_2d_joint(joint_img_ori, img_shape[1], self.human36_flip_pairs)
                joint_img_ori = joint_img_ori[:, :2]
                joint_img_ori = self.normalize_screen_coordinates(joint_img_ori, w=img_shape[1], h=img_shape[0])
            if not cfg.DATASET.use_gt_input:
                joint_img = self.replace_joint_img_wo_bbox(single_idx, img_id, joint_img, img_name, w=img_shape[1], h=img_shape[0])
            if flip:
                joint_img = flip_2d_joint(joint_img, img_shape[1], self.flip_pairs)
            
            joint_img = joint_img[:, :2]
            joint_img = self.normalize_screen_coordinates(joint_img, w=img_shape[1], h=img_shape[0])
            joint_img = np.array(joint_img, dtype=np.float32)
            joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)
            joint_smpl_cam = j3d_processing(joint_smpl_cam, rot, flip, self.smpl_flip_pairs)
            joint_cam_h36m = j3d_processing(joint_cam_h36m, rot, flip, self.human36_flip_pairs)

            joint_imgs.append(joint_img.reshape(1, len(joint_img), 2))
            joint_smpl_coors.append(joint_smpl_cam.reshape(1, 24, 3))
            if cfg.MODEL.name == 'pose2mesh_net':
                reg_joint_valid = np.ones((1, len(joint_cam_h36m), 1), dtype=np.float32)
                lift_joint_valid = np.ones((1, len(joint_cam), 1), dtype=np.float32)
                        
                joint_cams.append(joint_cam.reshape(1, len(joint_cam), 3))
                joint_cam_h36ms.append(joint_cam_h36m.reshape(1, len(joint_cam_h36m), 3))
                joint_img_oris.append(joint_img_ori.reshape(1, len(joint_img_ori), 2))
                pose_params.append(pose_param.reshape(1, -1))
                shape_params.append(shape_param.reshape(1, -1))
                
                lift_joint_valids.append(lift_joint_valid)
                reg_joint_valids.append(reg_joint_valid)

            elif cfg.MODEL.name == 'posenet': #  and num == int(self.seqlen / 2)
                joint_cams.append(joint_cam.reshape(1, len(joint_cam), 3))
                joint_valid = np.ones((1, len(joint_cam), 1), dtype=np.float32)
                lift_joint_valids.append(joint_valid)
        
        joint_imgs = np.concatenate(joint_imgs)
        if cfg.MODEL.name == 'pose2mesh_net':
            joint_cams, joint_cam_h36ms, joint_img_oris = np.concatenate(joint_cams), np.concatenate(joint_cam_h36ms), np.concatenate(joint_img_oris)
            pose_params, shape_params, joint_smpl_coors = np.concatenate(pose_params), np.concatenate(shape_params), np.concatenate(joint_smpl_coors)
            lift_joint_valids, reg_joint_valids = np.concatenate(lift_joint_valids), np.concatenate(reg_joint_valids)
            inputs = {'pose2d': joint_imgs}
            targets = {'lift_pose3d': joint_cams, 'reg_pose3d': joint_cam_h36ms, 'joint_img_ori': joint_img_oris, 'pose': pose_params, 'shape': shape_params, 'gt_joints24': joint_smpl_coors}
            meta = {'lift_pose3d_valid': lift_joint_valids, 'reg_pose3d_valid': reg_joint_valids}

            return inputs, targets, meta
        
        elif cfg.MODEL.name == 'posenet':
            joint_cams, lift_joint_valids = np.concatenate(joint_cams), np.concatenate(lift_joint_valids)
            joint_smpl_coors = np.concatenate(joint_smpl_coors)
            return joint_imgs, joint_cams, lift_joint_valids, joint_smpl_coors

    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def replace_joint_img_wo_bbox(self, idx, img_id, joint_img, img_name, w, h):
        if self.input_joint_name == 'coco':
            joint_img_coco = joint_img
            if self.data_split == 'train':
                det_data = self.datalist_pose2d_det_train[idx]
                det_name = self.datalist_pose2d_det_name_train[idx]
                assert img_name == det_name, f"check: {img_name} / {det_name}"
                joint_img_coco = det_data[:, :2].copy()
                return joint_img_coco
            else:
                det_data = self.datalist_pose2d_det[idx]
                joint_img_coco = det_data[:, :2].copy()
                return joint_img_coco
        if self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                det_data = self.datalist_pose2d_det_train[idx]
                det_name = self.datalist_pose2d_det_name_train[idx]
                assert img_name == det_name, f"check: {img_name} / {det_name}"
                joint_img_h36m = det_data[:, :2].copy()
                # joint_img_h36m[:, :2] = self.normalize_screen_coordinates(joint_img_h36m[:, :2].copy(), w=w, h=h)
                return joint_img_h36m
            else:
                det_data = self.datalist_pose2d_det[idx]
                det_name = self.datalist_pose2d_det_name[idx]
                assert img_name == det_name, f"check: {img_name} / {det_name}"
                joint_img_h36m = det_data[:, :2].copy()
                # joint_img_h36m[:, :2] = self.normalize_screen_coordinates(joint_img_h36m[:, :2].copy(), w=w, h=h)
                return joint_img_h36m
    
    def replace_joint_img(self, idx, img_id, joint_img, bbox, trans, img_name):
        if self.input_joint_name == 'coco':
            joint_img_coco = joint_img
            if self.data_split == 'train':
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                pt1 = affine_transform(np.array([xmin, ymin]), trans)
                pt2 = affine_transform(np.array([xmax, ymin]), trans)
                pt3 = affine_transform(np.array([xmax, ymax]), trans)
                area = math.sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2)) * math.sqrt(
                    pow(pt3[0] - pt2[0], 2) + pow(pt3[1] - pt2[1], 2))
                joint_img_coco[:17, :] = synthesize_pose(joint_img_coco[:17, :], area, num_overlap=0)
                return joint_img_coco
            else:
                joint_img_coco = self.datalist_pose2d_det[img_id]
                joint_img_coco = self.add_pelvis_and_neck(joint_img_coco)
                for i in range(self.coco_joint_num):
                    joint_img_coco[i, :2] = affine_transform(joint_img_coco[i, :2].copy(), trans)
                return joint_img_coco

        elif self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                # joint_syn_error = (self.generate_syn_error() / 256) * np.array(
                #     [cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]], dtype=np.float32)
                # joint_img_h36m = joint_img_h36m[:, :2] + joint_syn_error
                # return joint_img_h36m
                det_data = self.datalist_pose2d_det_train[idx]
                assert img_name == det_data['img_name'], f"check: {img_name} / {det_data['img_name']}"
                joint_img_h36m = det_data['pose2d'][:, :2].copy()
                for i in range(self.human36_joint_num):
                    joint_img_h36m[i, :2] = affine_transform(joint_img_h36m[i, :2].copy(), trans)
                return joint_img_h36m
            else:
                det_data = self.datalist_pose2d_det[idx]
                assert img_name == det_data['img_name'], f"check: {img_name} / {det_data['img_name']}"
                joint_img_h36m = det_data['pose2d'][:, :2].copy()
                for i in range(self.human36_joint_num):
                    joint_img_h36m[i, :2] = affine_transform(joint_img_h36m[i, :2].copy(), trans)
                return joint_img_h36m
            
    def compute_vertex_err(self, pred_verts, poses=None, shapes=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """
        from models.smpl_vibe import SMPL_MODEL_DIR
        from models.smpl_vibe import SMPL
        
        device = 'cpu'
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1, # target_theta.shape[0],
        ).to(device)
        
        target_verts = []
        b_ = torch.split(shapes, 5000)[0]
        p_ = torch.split(poses, 5000)[0]
        
        for b,p in zip(b_,p_):
            b = b.view(1, -1)
            p = p.view(1, -1)
            b, p = b.detach().cpu(), p.detach().cpu()
            output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
            vert = output.vertices.detach().cpu().numpy()
            joint = output.joints.detach().cpu().numpy()
            vert = vert - joint[:, 39]
            vert *= 1000
            target_verts.append(vert)

        target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        pred_verts = pred_verts.detach().cpu().numpy()
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
        error_per_vert = np.mean(error_per_vert, axis=1)
        error_per_vert = np.mean(error_per_vert)
        return error_per_vert, target_verts
            
    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error
    
    def compute_joint_err_h36m(self, pred_joint, target_joint):
        # root align joint, h36m joint set
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()
        
        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]

        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_acc_err(self, pred_joint, target_joint):
        # root align joint
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()
        
        pred_joint, target_joint = pred_joint[:, :, self.human36_eval_joint, :], target_joint[:, :, self.human36_eval_joint, :]
        
        accel_gt = target_joint[:, :-2] - 2 * target_joint[:, 1:-1] + target_joint[:, 2:]
        accel_pred = pred_joint[:, :-2] - 2 * pred_joint[:, 1:-1] + pred_joint[:, 2:]
        
        normed = np.linalg.norm(accel_pred - accel_gt, axis=3)
        
        new_vis = np.ones(len(normed), dtype=bool)
        acc_error = np.mean(normed[new_vis], axis=1)
        acc_error = np.mean(acc_error)

        return acc_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:, :1, :]
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]
        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate_joint(self, outs):
        logger.info('Evaluation start...')
        logger.info(len(outs))
        logger.info(len(self.vid_indices) * 16)
        sample_num = len(outs)        
        
        mpjpe = np.zeros((sample_num, len(self.human36_eval_joint)))
        pampjpe = np.zeros((sample_num, len(self.human36_eval_joint)))
        
        pred_j3ds_h36m = []  # acc error for each sequence
        gt_j3ds_h36m = []  # acc error for each sequence
        acc_error_h36m = 0.0
        last_seq_name = None
        
        for i in range(sample_num):
            out = outs[i]
            pose_coord_out, pose_coord_gt = out['joint_coord'], out['joint_coord_target']
            # root joint alignment
            pose_coord_out, pose_coord_gt = pose_coord_out - pose_coord_out[:1], pose_coord_gt - pose_coord_gt[:1]
            # sample eval joitns
            pose_coord_out, pose_coord_gt = pose_coord_out[self.human36_eval_joint, :], pose_coord_gt[self.human36_eval_joint, :]
            # pose error calculate
            mpjpe[i] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            
            pred_j3ds_h36m.append(pose_coord_out); gt_j3ds_h36m.append(pose_coord_gt)
        
            # perform rigid alignment
            pose_coord_out = rigid_align(pose_coord_out, pose_coord_gt)
            pampjpe[i] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
                
        # total pose error
        tot_err = np.mean(mpjpe)
        eval_summary = 'MPJPE (mm)    >> tot: %.2f' % (tot_err)
        logger.info(eval_summary)

        tot_err = np.mean(pampjpe)
        eval_summary = 'PA-MPJPE (mm) >> tot: %.2f' % (tot_err)
        logger.info(eval_summary)
        
        accel_error = []
        pred_j3ds = np.array(pred_j3ds_h36m); target_j3ds = np.array(gt_j3ds_h36m)
        for vid_idx in range(len(self.vid_indices)):
            pred, gt = pred_j3ds[(vid_idx * self.seqlen):(vid_idx * self.seqlen + self.seqlen)], target_j3ds[(vid_idx * self.seqlen):(vid_idx * self.seqlen + self.seqlen)]
            vid_acc_err = compute_error_accel(gt, pred)
            vid_acc_err = np.mean(vid_acc_err)
            accel_error.append(vid_acc_err)

        accel_error = np.mean(accel_error)
        acc_eval_summary = ('H36M acc error >> tot: %.2f\n ' % accel_error)
        logger.info(acc_eval_summary)
        

    def evaluate(self, outs):
        logger.info('Evaluation start...')
        sample_num = len(outs)

        sample_num_new = sample_num

        logger.info(sample_num_new)

        # eval H36M joints
        pose_error_h36m = np.zeros((sample_num_new, len(self.human36_eval_joint)))  # pose error
        pose_pa_error_h36m = np.zeros((sample_num_new, len(self.human36_eval_joint)))  # pose erro
        
        pred_j3ds_h36m = []  # acc error for each sequence
        gt_j3ds_h36m = []  # acc error for each sequence
        acc_error_h36m = 0.0
        
        # eval SMPL joints and mesh vertices
        pose_error = np.zeros((sample_num_new, self.smpl_joint_num))  # pose err
        mesh_error = np.zeros((sample_num_new, self.smpl_vertex_num))  # mesh error

        last_seq_name = None
        for n in range(sample_num):
            out = outs[n]
            # render materials
            img_path = self.img_paths[n]
            obj_name = '_'.join(img_path.split('/')[-2:])[:-4]

            # root joint alignment
            mesh_coord_out, mesh_coord_gt = out['mesh_coord'], out['mesh_coord_target']
            joint_coord_out, joint_coord_gt = np.dot(self.joint_regressor_smpl, mesh_coord_out), np.dot(self.joint_regressor_smpl, mesh_coord_gt)
            mesh_coord_out = mesh_coord_out - joint_coord_out[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            mesh_coord_gt = mesh_coord_gt - joint_coord_gt[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            pose_coord_out = joint_coord_out - joint_coord_out[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            pose_coord_gt = joint_coord_gt - joint_coord_gt[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]

            # pose error calculate
            pose_error[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))

            # mesh error calculate
            mesh_error[n] = np.sqrt(np.sum((mesh_coord_out - mesh_coord_gt) ** 2, 1))

            # pose error of h36m calculate
            pose_coord_out_h36m = np.dot(self.joint_regressor_human36, mesh_coord_out)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.human36_root_joint_idx]
            pose_coord_out_h36m = pose_coord_out_h36m[self.human36_eval_joint, :]
            pose_coord_gt_h36m = out['gt_pose']
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.human36_root_joint_idx]
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.human36_eval_joint, :]
            
            pred_j3ds_h36m.append(pose_coord_out_h36m); gt_j3ds_h36m.append(pose_coord_gt_h36m)
                
            pose_error_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1))
            pose_coord_out_h36m = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m) # perform rigid alignment
            pose_pa_error_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1))

            vis = cfg.TEST.vis
            if vis and (n % 500 == 0):
                mesh_to_save = mesh_coord_out / 1000
                obj_path = osp.join(cfg.vis_dir, f'{obj_name}.obj')
                save_obj(mesh_to_save, self.mesh_model.face, obj_path)

        accel_error = []
        pred_j3ds = np.array(pred_j3ds_h36m); target_j3ds = np.array(gt_j3ds_h36m)
        for vid_idx in range(len(self.vid_indices)):
            pred, gt = pred_j3ds[(vid_idx * self.seqlen):(vid_idx * self.seqlen + self.seqlen)], target_j3ds[(vid_idx * self.seqlen):(vid_idx * self.seqlen + self.seqlen)]
            vid_acc_err = compute_error_accel(gt, pred)
            vid_acc_err = np.mean(vid_acc_err)
            accel_error.append(vid_acc_err)
        accel_error = np.mean(accel_error)
        acc_eval_summary = ('H36M acc error >> tot: %.2f\n ' % accel_error)
        logger.info(acc_eval_summary)
            
        # total pose error (H36M joint set)
        tot_err = np.mean(pose_error_h36m)
        metric = 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' H36M pose error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        logger.info(eval_summary)

        tot_err = np.mean(pose_pa_error_h36m)
        metric = 'PA-MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' H36M pose error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        logger.info(eval_summary)

        # total pose error (SMPL joint set)
        tot_err = np.mean(pose_error)
        metric = 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' SMPL pose error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        logger.info(eval_summary)

        # total mesh error
        tot_err = np.mean(mesh_error)
        metric = 'MPVPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' SMPL mesh error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        logger.info(eval_summary)

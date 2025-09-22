import os.path as osp
import numpy as np
import cv2
import math
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

import Human36M.dataset, MPII3D.dataset, PW3D.dataset, COCO.dataset, MPII.dataset
import models
from multiple_datasets import MultipleDatasets
from core.loss import get_loss
from core.config import cfg
from display_utils import display_model
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, stop, lr_check, save_obj
from vis import vis_2d_pose, vis_3d_pose

from torchsummary import summary
from thop import profile
from thop import clever_format
from geometry import batch_rodrigues


import wandb
'wandb'

logger = logging.getLogger(__name__)

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000., # 5000mm = 5m
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d

def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    logger.info(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        dataset = eval(f'{name}.dataset')(dataset_split.lower(), args=args)
        logger.info("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=batch_per_dataset * len(dataset_names), shuffle=cfg[dataset_split].shuffle,
                                     num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator


def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    J_regressor = eval(f'torch.Tensor(main_dataset.joint_regressor_{cfg.DATASET.input_joint_set})') # .cuda()
    if is_train or load_dir:
        logger.info(f"==> Preparing {cfg.MODEL.name} MODEL...")
        if cfg.MODEL.name == 'LINK':
            model = models.LINK.get_model(num_joint=main_dataset.joint_num, embed_dim=256, depth=5, J_regressor=J_regressor) # 
        elif cfg.MODEL.name == 'posenet':
            model = models.posenet.get_model(num_joint=main_dataset.joint_num, embed_dim=256, depth=5, J_regressor=J_regressor, pretrained=cfg.MODEL.posenet_pretrained)
        logger.info('# of model parameters: {}'.format(count_parameters(model)))

    if is_train:
        criterion = get_loss(faces=main_dataset.mesh_model.face)
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        logger.info('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir, pick_best=(cfg.MODEL.name == 'posenet'))
        model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']
            # update lr_scheduler
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            logger.info('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


class Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')
        self.J_regressor_smpl = eval(f'torch.Tensor(self.main_dataset.joint_regressor_smpl).cuda()')
        
        self.model = self.model.cuda()

        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='light_pmce_pose2mesh/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch): # 训练模块
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_pose = inputs['pose2d'].cuda()
            gt_lift3dpose, gt_reg3dpose = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda()
            val_lift3dpose, val_reg3dpose = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda()
            target_pose2d = targets['joint_img_ori'].cuda()
            real_pose, real_shape = targets['pose'].cuda(), targets['shape'].cuda()
            gt_joints24 = targets['gt_joints24'].cuda()
            real_shape = real_shape.squeeze(1)
            
            gt_reg3dpose = gt_reg3dpose.view(-1, gt_reg3dpose.shape[2], gt_reg3dpose.shape[3])
            target_pose2d = target_pose2d.view(-1, target_pose2d.shape[2], target_pose2d.shape[3])
            gt_joints24 = gt_joints24.view(-1, gt_joints24.shape[2], gt_joints24.shape[3])
            real_pose = real_pose.view(-1, real_pose.shape[2])
            real_shape = real_shape.view(-1, real_shape.shape[2])
            val_reg3dpose = val_reg3dpose.view(-1, val_reg3dpose.shape[2], val_reg3dpose.shape[3])
            

            pred_mesh, pred_24joint, pred_camera, pred_theta_mats, pred_shape = self.model(input_pose) 
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh * 1000)
            pred_pose_2d = projection(pred_pose * 0.001, pred_camera)
            pred_24joint = pred_24joint * 1000
                           
            mseh2joint3d_loss = self.joint_weight * self.loss[3](pred_pose,  gt_reg3dpose, val_reg3dpose)
            reproj_pose_2d = self.joint_weight * self.loss[5](pred_pose_2d, target_pose2d, val_reg3dpose)
            joint24_loss = self.joint_weight * self.loss[7](pred_24joint, gt_joints24, True)
            pose_para_loss, shape_para_loss = self.loss[9](pred_theta_mats, pred_shape, real_pose, real_shape)
            pose_para_loss, shape_para_loss = pose_para_loss * 2.0, shape_para_loss * 0.5
            loss = mseh2joint3d_loss + reproj_pose_2d + joint24_loss + pose_para_loss + shape_para_loss # loss1 + 
 
            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_mseh2joint3d_loss = mseh2joint3d_loss.detach() 
                wandb_reproj_pose_2d = reproj_pose_2d.detach()
                wandb_joint24_loss = joint24_loss.detach() 
                wandb_pose_para_loss = pose_para_loss.detach()
                wandb_shape_para_loss = shape_para_loss.detach()            
                wandb.log(
                    {
                        'train_loss/mseh2joint3d_loss': wandb_mseh2joint3d_loss,
                        'train_loss/reproj_pose_2d': wandb_reproj_pose_2d,
                        'train_loss/joint24_loss': wandb_joint24_loss,
                        'train_loss/pose_para_loss': wandb_pose_para_loss,
                        'train_loss/shape_para_loss': wandb_shape_para_loss
                    }
                )

            if i % self.print_freq == 0:
                mseh2joint3d_loss = mseh2joint3d_loss.detach()
                reproj_pose_2d, joint24_loss = reproj_pose_2d.detach(), joint24_loss.detach()
                pose_para_loss, shape_para_loss = pose_para_loss.detach(), shape_para_loss.detach()
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                                               f'mesh->3d joint loss: {mseh2joint3d_loss:.4f} '
                                               f'reproj loss: {reproj_pose_2d:.4f} '
                                               f'joint24 loss: {joint24_loss:.4f} '
                                               f'pose para loss: {pose_para_loss:.4f} '
                                               f'shape para loss: {shape_para_loss:.4f} '
                                               )
            

        self.loss_history.append(running_loss / len(batch_generator))

        logger.info(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class Tester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)

        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.val_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval() 

        surface_error = 0.0
        joint_error = 0.0

        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose = inputs['pose2d'].cuda()
                gt_pose3d = targets['reg_pose3d'].cuda()
                joint_3d = targets['lift_pose3d'].cuda()
                real_pose, real_shape = targets['pose'].cuda(), targets['shape'].cuda()
                real_pose, real_shape = real_pose.view(-1, real_pose.shape[2]), real_shape.view(-1, real_shape.shape[2])

                pred_mesh, pred_24joint, pred_camera, pred_theta_mats, pred_shape = self.model(input_pose)
                pred_mesh = pred_mesh * 1000

                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
                gt_pose3d = gt_pose3d.view(-1, gt_pose3d.shape[2], gt_pose3d.shape[3])
                joint_3d = joint_3d.view(-1, joint_3d.shape[2], joint_3d.shape[3])
                pred_mesh = pred_mesh - pred_pose[:, :1, :]
                

                j_error = self.val_dataset.compute_joint_err_h36m(pred_pose, gt_pose3d)
                s_error, gt_mesh = self.val_dataset.compute_vertex_err(pred_mesh, real_pose, real_shape)
                
                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => surface error: {s_error:.4f}, joint error: {j_error:.4f}')

                joint_error += j_error
                surface_error += s_error

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), gt_mesh
                    joint_3d = joint_3d.detach().cpu().numpy()
                    for j in range(len(pred_mesh)):
                        out = {}
                        out['mesh_coord'], out['mesh_coord_target'] = pred_mesh[j], target_mesh[j]
                        out['gt_pose'] = joint_3d[j]
                        result.append(out)

            self.surface_error = surface_error / len(self.val_loader)
            self.joint_error = joint_error / len(self.val_loader)
            logger.info(f'{eval_prefix} MPVPE: {self.surface_error:.2f}, MPJPE: {self.joint_error:.2f}')

            if cfg.TRAIN.wandb:
                wandb_joint_error = self.joint_error
                wandb_verts_error = self.surface_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_joint_error,
                        'error/MPVPE': wandb_verts_error
                    }
                )

            # Final Evaluation
            if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                self.val_dataset.evaluate(result)



class LiftTrainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history \
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.loss, self.loss1, self.accloss, self.accloss_smpl = self.loss[0], self.loss[3], self.loss[10], self.loss[11]
        self.main_dataset = self.dataset_list[0]
        self.num_joint = self.main_dataset.joint_num
        self.print_freq = cfg.TRAIN.print_freq

        self.model = self.model.cuda()

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='light_pmce_posenet/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (img_joint, cam_joint, joint_valid, gt_joints24) in enumerate(batch_generator): # 
            img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
            joint_valid = joint_valid.cuda().float()
            gt_joints24 = gt_joints24.cuda().float()

            pred_joint, pred_joints24 = self.model(img_joint)
            pred_joint = pred_joint.view(-1, self.num_joint, 3)
            cam_joint = cam_joint.view(-1, self.num_joint, 3)
            gt_joints24 = gt_joints24.view(-1, 24, 3)
            joint_valid = joint_valid.view(-1, self.num_joint, 1)

            loss = 1/cfg.DATASET.seqlen * self.loss(pred_joint, cam_joint, joint_valid)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_loss = loss.detach()
                wandb.log(
                    {
                        'train_loss/total_loss': wandb_loss
                    }
                )

            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                f'total loss: {loss.detach():.4f} ')

        self.loss_history.append(running_loss / len(self.batch_generator))

        logger.info(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class LiftTester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.val_dataset = self.val_dataset[0]
        self.val_loader = self.val_loader[0]

        self.num_joint = self.val_dataset.joint_num
        self.print_freq = cfg.TRAIN.print_freq

        if self.model:
            self.model = self.model.cuda()
            input_2d = torch.randn(1, 16, 19, 2).cuda()
            flops, params = profile(self.model, inputs=(input_2d, ))
            print('flops: ', flops, 'params: ', params)
            flops, params = clever_format([flops, params], "%.3f")
            print(flops, params)

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9
        self.joint_error_smpl = 9999.9
        self.acc_error = 9999.9
        self.acc_error_smpl = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        

        result = []
        joint_error = 0.0
        joint_error_smpl = 0.0
        acc_error = 0.0
        acc_error_smpl = 0.0
        
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (img_joint, cam_joint, _, gt_joints24) in enumerate(loader):
                img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
                gt_joints24 = gt_joints24.cuda().float()

                pred_joint, pred_joint_smpl = self.model(img_joint)
                pred_joint = pred_joint.view(-1, self.num_joint, 3)
                cam_joint = cam_joint.view(-1, self.num_joint, 3)

                mpjpe = self.val_dataset.compute_joint_err(pred_joint, cam_joint)
                joint_error += mpjpe
                
                pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
                cam_joint = cam_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
                acc_err = self.val_dataset.compute_acc_err(pred_joint, cam_joint)
                acc_error += acc_err

                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => joint error: {mpjpe:.4f}, acc error: {acc_err:.4f}')

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_joint, target_joint = pred_joint.view(-1, self.num_joint, 3).detach().cpu().numpy(), cam_joint.view(-1, self.num_joint, 3).detach().cpu().numpy()
                    for j in range(len(pred_joint)):
                        out = {}
                        out['joint_coord'], out['joint_coord_target'] = pred_joint[j], target_joint[j]
                        result.append(out)

        self.joint_error = joint_error / len(self.val_loader)
        self.acc_error = acc_error / len(self.val_loader)
        
        
        logger.info(f'{eval_prefix} MPJPE: {self.joint_error:.4f} ACC_ERROR: {self.acc_error:.4f}')

        if cfg.TRAIN.wandb:
                wandb_error = self.joint_error
                wandb_acc_error = self.acc_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_error,
                        'error/ACC_ERROR': wandb_acc_error,
                    }
                )

        # Final Evaluation
        if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
            self.val_dataset.evaluate_joint(result)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import cfg
from funcs_utils import stop
from geometry import batch_rodrigues, quat_to_rotmat


class CoordLoss(nn.Module):
    def __init__(self, has_valid=False):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, target_valid):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target)

        return loss
    
class ACCLoss(nn.Module):
    def __init__(self):
        super(ACCLoss, self).__init__()

    def forward(self, pred, target):
        accel_gt = target[:, :-2] - 2 * target[:, 1:-1] + target[:, 2:]
        accel_pred = pred[:, :-2] - 2 * pred[:, 1:-1] + pred[:, 2:]
        
        return torch.mean(torch.norm(accel_pred - accel_gt, dim=len(accel_gt.shape)-1))
    
class VELLoss(nn.Module):
    def __init__(self):
        super(VELLoss, self).__init__()

    def forward(self, pred, target):
        vel_gt = target[:, 1:] - target[:, :-1]
        vel_pred = pred[:, 1:] - pred[:, :-1]
        
        return torch.mean(torch.norm(vel_pred - vel_gt, dim=len(vel_gt.shape)-1))
        
    
class SMPL_PARM(nn.Module):
    def __init__(self):
        super(SMPL_PARM, self).__init__()
        
        self.criterion = nn.MSELoss()
        
    def forward(self, pred_pose, pred_beta, gt_pose, gt_beta):
        pred_pose, pred_beta = pred_pose.to(torch.float32), pred_beta.to(torch.float32)
        gt_pose, gt_beta = gt_pose.to(torch.float32), gt_beta.to(torch.float32)
        pred_rotmat_valid = quat_to_rotmat(pred_pose.reshape(-1, 4)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_beta
        gt_betas_valid = gt_beta
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas


class LaplacianLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = 6890  # SMPL
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= (laplacian[i, i] + 1e-8)

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda().float())

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.cat([torch.matmul(self.laplacian, x[i])[None, :, :] for i in range(batch_size)], 0)

        x = x.pow(2).sum(2)
        if self.average:
            return x.sum() / batch_size
        else:
            return x.mean()


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()
    
def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))


def get_loss(faces):
    loss = CoordLoss(has_valid=True), NormalVectorLoss(faces), EdgeLengthLoss(faces), CoordLoss(has_valid=True), CoordLoss(has_valid=True), \
           CoordLoss(has_valid=True), CoordLoss(has_valid=True), \
           CoordLoss(has_valid=False), CoordLoss(has_valid=False), \
           SMPL_PARM(), ACCLoss(), ACCLoss(), VELLoss()

    return loss

import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import meshnet, posenet


class LINK(nn.Module):
    def __init__(self, num_joint, embed_dim, depth, J_regressor):
        super(LINK, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = posenet.get_model(num_joint, embed_dim, depth, J_regressor, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose2mesh = meshnet.get_model(num_joint, embed_dim)

    def forward(self, pose2d):
        pose3d, _ = self.pose_lifter(pose2d)
        pose3d = pose3d.reshape(-1, self.num_joint, 3)
        pred_vertices, pred_24joint, pred_camera, pred_theta_mats, pred_shape = self.pose2mesh(pose3d / 1000)

        return pred_vertices, pred_24joint, pred_camera, pred_theta_mats, pred_shape


def get_model(num_joint, embed_dim, depth, J_regressor):
    model = LINK(num_joint, embed_dim, depth, J_regressor)

    return model



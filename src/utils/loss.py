import torch
import torch.nn as nn
from utils.geometry_utils import unit_normals


class L1Loss(nn.Module):
    def __init__(self, invalid_mask_weight=0.0):
        super(L1Loss, self).__init__()
        self.name = 'L1'
        self.invalid_mask_weight = invalid_mask_weight

    def forward(self, pred, target, mask):
        loss = nn.functional.l1_loss(pred[mask], target[mask])
        if self.invalid_mask_weight > 0.0:
            invalid_mask = ~mask
            if invalid_mask.sum() > 0:
                invalid_loss = nn.functional.l1_loss(pred[invalid_mask], target[invalid_mask])
                loss = loss + self.invalid_mask_weight * invalid_loss
        return loss



class GradL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'GradL1'

    def grad(self, x):
        dx = x[..., :-1, 1:] - x[..., :-1, :-1]
        dy = x[..., 1:, :-1] - x[..., :-1, :-1]
        return dx, dy

    def grad_mask(self, mask):
        return (mask[..., :-1, :-1] & mask[..., :-1, 1:] &
                mask[..., 1:, :-1] & mask[..., 1:, 1:])

    def forward(self, pred, target, mask):
        dx_p, dy_p = self.grad(pred)
        dx_t, dy_t = self.grad(target)
        mask_g = self.grad_mask(mask)

        loss_x = nn.functional.l1_loss(dx_p[mask_g], dx_t[mask_g], reduction='mean')
        loss_y = nn.functional.l1_loss(dy_p[mask_g], dy_t[mask_g], reduction='mean')

        return 0.5 * (loss_x + loss_y)


class CosineNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CosineNormalLoss"

    def forward(self, pred: torch.Tensor,
                      target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, "pred and target must have same shape"

        pred = unit_normals(pred)
        target = unit_normals(target)

        dot = (pred * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        cos_term = 1.0 - dot

        if mask is not None:
            loss = cos_term[mask].mean()
        else:
            loss = cos_term.mean()
        return loss

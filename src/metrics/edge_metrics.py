# @Vukasin Bozic 2026
# References for the implementation of edge metrics:
# - Koch et al., "Evaluation of CNN-based Single-Image Depth Estimation Methods",
#   https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Koch_Evaluation_of_CNN-based_Single-Image_Depth_Estimation_Methods_ECCVW_2018_paper.pdf
# - Hu et al., "Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries"
# - Pano3D implementation: https://vcl3d.github.io/Pano3D/


import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

def compute_sobel_edges(
    image, threshold=0.5, dilation_kernel=3
):
    image = image[None, ...][None, ...]

    padded_image = F.pad(
        image,
        (dilation_kernel, dilation_kernel, dilation_kernel, dilation_kernel),
        mode="replicate",
    )

    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=image.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=image.device
    ).view(1, 1, 3, 3)

    edges_x = F.conv2d(padded_image, sobel_x, padding=1)
    edges_y = F.conv2d(padded_image, sobel_y, padding=1)

    edges = torch.sqrt(edges_x**2 + edges_y**2)

    if threshold == None:
        threshold = edges.mean()
    edges = (edges > threshold).float()

    dilated_edges = F.max_pool2d(
        edges, kernel_size=dilation_kernel, stride=1, padding=dilation_kernel // 2
    )

    dilated_edges = dilated_edges[
        :, :, dilation_kernel:-dilation_kernel, dilation_kernel:-dilation_kernel
    ]

    return dilated_edges.squeeze(0).squeeze(0).cpu().numpy()


def compute_skimage_edges_sobel(depth, mask, thresh=0.01):

    depth_normalized = depth.clone()
    depth_min = torch.min(depth_normalized[mask])
    depth_max = torch.max(depth_normalized[mask])
    depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)

    edges = compute_sobel_edges(
        depth_normalized, threshold=thresh
    )
    return edges


class MetricTracker:
    def __init__(self, tracked_metrics, max_depth, save_error_list=False):
        self.tracked_metrics = tracked_metrics
        self.metrics_sum = {metric: 0.0 for metric in tracked_metrics}
        self.save_error_list = save_error_list
        self.error_list = {metric: [] for metric in tracked_metrics}
        self.max_depth = max_depth

    def edge_dbe_completeness(self, edges_est, edges_gt, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            edges_gt = edges_gt * valid_mask
            edges_est = edges_est * valid_mask

        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        mask_D_gt = D_gt < self.max_depth
        E_fin_est_filt = edges_est * mask_D_gt

        if E_fin_est_filt.sum() == 0:
            dbe_com = torch.tensor([self.max_depth])
        else:
            ch1 = D_gt * edges_est
            ch1[ch1 > self.max_depth] = self.max_depth
            ch2 = D_est * edges_gt
            ch2[ch2 > self.max_depth] = self.max_depth
            res = ch1 + ch2
            dbe_com = np.nansum(res) / (
                np.nansum(edges_est) + np.nansum(edges_gt)
            )

        return dbe_com


    def edge_dbe_accuracy(self, edges_est, edges_gt, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            edges_gt = edges_gt * valid_mask
            edges_est = edges_est * valid_mask

        D_gt = ndimage.distance_transform_edt(1 - edges_gt)

        mask_D_gt = D_gt < self.max_depth
        E_fin_est_filt = edges_est * mask_D_gt

        if E_fin_est_filt.sum() == 0:
            dbe_acc = torch.tensor([self.max_depth])
        else:
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(
                E_fin_est_filt
            )

        return dbe_acc


    def edge_precision(self, edges_est, edges_gt, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            edges_gt = edges_gt * valid_mask
            edges_est = edges_est * valid_mask

        edges_gt = torch.from_numpy(edges_gt)
        edges_est = torch.from_numpy(edges_est)

        if torch.where(edges_gt == 1, 1, 0).sum() == 0:
            prec_ = torch.tensor([1])
        else:
            false_positives = torch.where((edges_gt != 1) & (edges_est == 1), 1, 0).sum()
            true_positives = torch.where((edges_gt == 1) & (edges_est == 1), 1, 0).sum()
            if torch.where(edges_est == 1, 1, 0).sum() == 0:
                prec_ = torch.tensor([0])
            else:
                prec_ = true_positives / (true_positives + false_positives)

        return prec_


    def edge_recall(self, edges_est, edges_gt, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            edges_gt = edges_gt * valid_mask
            edges_est = edges_est * valid_mask

        edges_gt = torch.from_numpy(edges_gt)
        edges_est = torch.from_numpy(edges_est)

        if torch.where(edges_gt == 1, 1, 0).sum() == 0:
            recall_ = torch.tensor([1])
        else:
            true_positives = torch.where((edges_gt == 1) & (edges_est == 1), 1, 0).sum()
            false_negatives = torch.where((edges_gt == 1) & (edges_est == 0), 1, 0).sum()
            recall_ = true_positives / (true_positives + false_negatives)

        return recall_


    def update(self, depth_pred, depth_gt, valid_mask, id):
        for metric in self.tracked_metrics:
            metric_fn = getattr(self, metric)
            pred_edges = compute_skimage_edges_sobel(depth_pred, valid_mask)
            gt_edges = compute_skimage_edges_sobel(depth_gt, valid_mask)
            value = metric_fn(pred_edges, gt_edges, valid_mask)
            self.metrics_sum[metric] += value.item()
            if self.save_error_list:
               self.error_list[metric].append((id, value.item()))


    def calculate_final(self, num_samples):
        final_metrics = {}
        for metric in self.tracked_metrics:
            final_metrics[metric] = self.metrics_sum[metric] / num_samples
            print(f"{metric}: {final_metrics[metric]:.4f}")
        return final_metrics, self.error_list
    


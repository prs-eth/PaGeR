import torch
from utils.utils import compute_scale_and_shift

def align_pred_gt(depth_pred, depth_gt, valid_mask, alignment_type):
    if alignment_type == "scale_and_shift":
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, valid_mask, fit_shift=True)
    elif alignment_type == "scale":
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, valid_mask, fit_shift=False)
        shift = 0.0
    elif alignment_type == "metric":
        scale = 1.0
        shift = 0.0
    else:
        raise ValueError(f"Unknown alignment type: {alignment_type}")
    
    depth_pred = scale * depth_pred + shift
    return depth_pred


class MetricTracker:
    def __init__(self, tracked_metrics, save_error_list=False):
        self.tracked_metrics = tracked_metrics
        self.metrics_sum = {metric: 0.0 for metric in tracked_metrics}
        self.erp_weights = None
        self.save_error_list = save_error_list
        self.error_list = {metric: [] for metric in tracked_metrics}

    def erp_cosine_weights(self, H, W, device=None, dtype=torch.float32):
        ii  = torch.arange(H, device=device, dtype=dtype) + 0.5
        lat = (ii / H) * torch.pi - 0.5 * torch.pi
        w_row = torch.cos(lat).clamp_min(0.0)

        w_row = w_row / w_row.mean().clamp_min(1e-8)

        w = w_row[:, None].expand(H, W)
        return w.unsqueeze(0)

    def abs_relative_difference(self, output, target, valid_mask=None):
        actual_output = output
        actual_target = target
        abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
        if valid_mask is not None:
            abs_relative_diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = output.shape[-1] * output.shape[-2]
        if abs_relative_diff.dim() == 4:
            abs_relative_diff = abs_relative_diff.permute(1, 0, 2, 3)
            n = n.sum()
            abs_relative_diff = torch.sum(self.erp_weights * abs_relative_diff, (-1, -2, -3)) / n
        else:
            abs_relative_diff = torch.sum(self.erp_weights * abs_relative_diff, (-1, -2)) / n
        return abs_relative_diff.mean()


    def squared_relative_difference(self, output, target, valid_mask=None):
        actual_output = output
        actual_target = target
        square_relative_diff = (
            torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
        )
        if valid_mask is not None:
            square_relative_diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = output.shape[-1] * output.shape[-2]
        square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
        return square_relative_diff.mean()


    def rmse_linear(self, output, target, valid_mask=None):
        actual_output = output
        actual_target = target
        diff = actual_output - actual_target
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = output.shape[-1] * output.shape[-2]
        diff2 = torch.pow(diff, 2)
        if diff.dim() == 4:
            diff2 = diff2.permute(1, 0, 2, 3)
            n = n.sum()
            mse = torch.sum(self.erp_weights * diff2, (-1, -2, -3)) / n
        else:
            mse = torch.sum(self.erp_weights * diff2, (-1, -2)) / n
        rmse = torch.sqrt(mse)
        return rmse.mean()


    def rmse_log(self, output, target, valid_mask=None):
        diff = torch.log(output) - torch.log(target)
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = output.shape[-1] * output.shape[-2]
        diff2 = torch.pow(diff, 2)
        mse = torch.sum(diff2, (-1, -2)) / n
        rmse = torch.sqrt(mse)
        return rmse.mean()


    def log10(self, output, target, valid_mask=None):
        if valid_mask is not None:
            diff = torch.abs(
                torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
            )
        else:
            diff = torch.abs(torch.log10(output) - torch.log10(target))
        return diff.mean()


    # adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
    def threshold_percentage(self, output, target, threshold_val, valid_mask=None):
        d1 = output / target
        d2 = target / output
        max_d1_d2 = torch.max(d1, d2)
        zero = torch.zeros(*output.shape, device=output.device)
        one = torch.ones(*output.shape, device=output.device)
        bit_mat = torch.where(max_d1_d2 < threshold_val, one, zero)
        if valid_mask is not None:
            bit_mat[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = output.shape[-1] * output.shape[-2]
        
        if bit_mat.dim() == 4:
            bit_mat = bit_mat.permute(1, 0, 2, 3)
            n = n.sum()
            count_mat = torch.sum(self.erp_weights * bit_mat, (-1, -2, -3))
        else:
            count_mat = torch.sum(self.erp_weights * bit_mat, (-1, -2))

        threshold_mat = count_mat / n
        return threshold_mat.mean()


    def delta1_acc(self, pred, gt, valid_mask):
        return self.threshold_percentage(pred, gt, 1.25, valid_mask)


    def delta2_acc(self, pred, gt, valid_mask):
        return self.threshold_percentage(pred, gt, 1.25**2, valid_mask)


    def delta3_acc(self, pred, gt, valid_mask):
        return self.threshold_percentage(pred, gt, 1.25**3, valid_mask)


    def i_rmse(self, output, target, valid_mask=None):
        output_inv = 1.0 / output
        target_inv = 1.0 / target
        diff = output_inv - target_inv
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = output.shape[-1] * output.shape[-2]
        diff2 = torch.pow(diff, 2)
        mse = torch.sum(diff2, (-1, -2)) / n
        rmse = torch.sqrt(mse)
        return rmse.mean()


    def silog_rmse(self, depth_pred, depth_gt, valid_mask=None):
        diff = torch.log(depth_pred) - torch.log(depth_gt)
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
        return loss


    def update(self, depth_pred, depth_gt, valid_mask, id):
        if self.erp_weights is None:
            H, W = depth_gt.shape[-2], depth_gt.shape[-1]
            self.erp_weights = torch.ones((1, H, W), device=depth_gt.device, dtype=depth_gt.dtype)


        for metric in self.tracked_metrics:
            metric_fn = getattr(self, metric)
            value = metric_fn(depth_pred, depth_gt, valid_mask)
            self.metrics_sum[metric] += value.item()
            if self.save_error_list:
                self.error_list[metric].append((id, value.item()))


    def calculate_final(self, num_samples):
        final_metrics = {}
        for metric in self.tracked_metrics:
            final_metrics[metric] = self.metrics_sum[metric] / num_samples
            print(f"{metric}: {final_metrics[metric]:.4f}")
        return final_metrics, self.error_list
    

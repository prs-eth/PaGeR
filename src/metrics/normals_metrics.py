import torch
import numpy as np

class MetricTracker:
    def __init__(self, tracked_metrics):
        self.tracked_metrics = tracked_metrics
        self.metrics_sum = {metric: 0.0 for metric in tracked_metrics}


    def compute_normals_metrics(gt, pred, mask, radians, full_region):
        dot_products = np.sum(pred * gt, axis=3, keepdims=True)
        angle_radians = np.arccos(np.clip(dot_products, -1.0, 1.0))

        if radians:
            E = angle_radians
        else:
            E = np.degrees(angle_radians)
            mask = mask.detach().cpu().numpy().transpose(0,2,3,1).astype(np.bool8)

            if full_region:
                E *= mask
            else:
                E = E[mask]


        return (np.mean(E), np.median(E),
                (np.mean(np.power(E,2))),
                np.sqrt(np.mean(np.power(E,2))),
                np.mean(E < 5) * 100,
                np.mean(E < 7.5) * 100,
                np.mean(E < 11.25) * 100,
                np.mean(E < 22.5 ) * 100,
                np.mean(E < 30   ) * 100)


    def update(self, pred, gt, valid_mask):

        dot_products = torch.sum(pred * gt, dim=0, keepdims=True)
        angle_radians = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        angle_degrees = torch.rad2deg(angle_radians)

        self.metrics_sum["mean"] += torch.mean(angle_degrees[valid_mask]).item()
        self.metrics_sum["median"] += torch.median(angle_degrees[valid_mask]).item()
        self.metrics_sum["mse"] += torch.mean(angle_degrees[valid_mask] ** 2).item()
        self.metrics_sum["delta_5"] += torch.mean((angle_degrees[valid_mask] < 5).float()).item()
        self.metrics_sum["delta_7.5"] += torch.mean((angle_degrees[valid_mask] < 7.5).float()).item()
        self.metrics_sum["delta_11.25"] += torch.mean((angle_degrees[valid_mask] < 11.25).float()).item()
        self.metrics_sum["delta_22.5"] += torch.mean((angle_degrees[valid_mask] < 22.5).float()).item()
        self.metrics_sum["delta_30"] += torch.mean((angle_degrees[valid_mask] < 30).float()).item()


    def calculate_final(self, num_samples):
        final_metrics = {}
        for metric in self.tracked_metrics:
            final_metrics[metric] = self.metrics_sum[metric] / num_samples
            print(f"{metric}: {final_metrics[metric]:.4f}")
        return final_metrics
    

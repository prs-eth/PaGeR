# @Vukasin Bozic 2026
# This file contains the modified version of Marigold's exponential LR scheduler. 
# https://github.com/prs-eth/Marigold/blob/main/src/util/lr_scheduler.py

# Author: Bingxin Ke

import numpy as np

class IterExponential:
    
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        self.total_length = total_iter_length
        self.effective_length = int(total_iter_length * (1 - warmup_steps))
        self.final_ratio = final_ratio
        self.warmup_steps = int(total_iter_length * warmup_steps)

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha
    

class IterConstant:
    
    def __init__(self, total_iter_length: int, warmup_steps: float = 0.0) -> None:
        self.total_length = int(total_iter_length)
        self.warmup_steps = int(total_iter_length * warmup_steps)

    def __call__(self, n_iter: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        if n_iter < self.warmup_steps:
            return float(n_iter + 1) / float(self.warmup_steps)
        return 1.0
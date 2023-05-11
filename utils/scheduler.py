import numpy as np


def get_lr(start_lr, end_lr, warmup_step, cur_step, last_iter):
    if cur_step < warmup_step:
        mul = cur_step / warmup_step
        return start_lr * mul
    elif cur_step <= last_iter:
        ratio = 0.5 * (1.0 + np.cos((cur_step - warmup_step) / (last_iter - warmup_step) * np.pi))
        return (start_lr - end_lr) * ratio + end_lr
    else:
        return end_lr

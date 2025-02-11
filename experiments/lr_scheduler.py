
import math
from abc import abstractmethod

from torch.optim import lr_scheduler


class BaseLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self) -> list[float]:
        """Returns the current learning rate for each parameter group."""
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self, **kwargs) -> None:
        """Reinitializes the learning rate scheduler."""
        raise NotImplementedError


class LinearWarmupCosineAnnealing(BaseLRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_until_step, max_lr, min_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_until_step = decay_until_step
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def compute_lr(step, warmup_steps, decay_until_step, max_lr, min_lr):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        if step > decay_until_step:
            return min_lr
        if warmup_steps <= step < decay_until_step:
            decay_ratio = (step - warmup_steps) / (decay_until_step - warmup_steps)
            assert 0.0 <= decay_ratio <= 1.0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)
        else:
            return min_lr

    def get_lr(self) -> list[float]:
        """Returns the current learning rate for each parameter group."""
        step = self.last_epoch
        return (
            self.compute_lr(step, self.warmup_steps, self.decay_until_step, self.max_lr, self.min_lr)
            for _ in self.optimizer.param_groups
        )


class CosineIncrementConstant():
    def __init__(self, max_lr, min_lr, start_step, end_step, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.end_step = end_step
        self.start_step = start_step

    def compute_lr(self, step):
        if step < self.start_step:
            return self.min_lr
        if step > self.end_step:
            return self.max_lr
        else:
            increment_ratio = (step - self.start_step) / (self.end_step - self.start_step)
            assert 0.0 <= increment_ratio <= 1.0
            coeff = 0.5 * (1.0 - math.cos(math.pi * increment_ratio))
            return self.min_lr + coeff * (self.max_lr - self.min_lr)
            
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

class Optimizer:
    def __init__(self, model_params, opt_type='SGD', lr=0.001, scheduler_type=None, scheduler_params=None, **kwargs):
        opt_type = opt_type.lower()
        if opt_type == 'sgd':
            self.optimizer = optim.SGD(model_params,
                                       lr=lr,
                                       momentum=kwargs.get('momentum', 0.9),
                                       weight_decay=kwargs.get('weight_decay', 0.0)
                                       )
        elif opt_type == 'adam':
            self.optimizer = optim.Adam(model_params,
                                        lr=lr,
                                        betas=kwargs.get('betas', (0.9, 0.999)),
                                        weight_decay=kwargs.get('weight_decay', 0.0)
                                        )
        elif opt_type == 'adamw':
            self.optimizer = optim.AdamW(model_params,
                                         lr=lr,
                                         betas=kwargs.get('betas', (0.9, 0.999)),
                                         weight_decay=kwargs.get('weight_decay', 0.0)
                                         )
        elif opt_type == 'rmsprop':
            self.optimizer = optim.RMSprop(model_params,
                                           lr=lr,
                                           momentum=kwargs.get('momentum', 0.0),
                                           weight_decay=kwargs.get('weight_decay', 0.0)
                                           )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        self.scheduler = None
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, **scheduler_params)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

"""
Optimizer and scheduler creation utilities
"""
import torch
import torch.optim as optim


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    if config.training.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch=None):
    """Create learning rate scheduler based on configuration"""
    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.min_lr
        )
    elif config.training.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )
    elif config.training.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.training.milestones,
            gamma=config.training.gamma
        )
    elif config.training.scheduler == "cosine_warmup" and steps_per_epoch:
        # Implement cosine annealing with warmup
        warmup_steps = config.training.warmup_epochs * steps_per_epoch
        total_steps = config.training.epochs * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    return scheduler
"""
Metrics tracking utilities for training and evaluation
"""
import torch
import numpy as np
from collections import defaultdict


class MetricsTracker:
    """Track and compute training metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        
    def update(self, loss, predictions, targets):
        self.losses.append(loss)
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
    def get_metrics(self):
        avg_loss = np.mean(self.losses)
        accuracy = np.mean(np.array(self.predictions) == np.array(self.targets))
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
    def log_metrics(self, metrics, prefix=''):
        for key, value in metrics.items():
            print(f"{prefix}{key}: {value:.4f}")


class MAEMetricsTracker:
    """Track and compute MAE training metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses = []
        self.total_samples = 0
        
    def update(self, loss, batch_size):
        self.losses.append(loss * batch_size)
        self.total_samples += batch_size
        
    def get_metrics(self):
        avg_loss = sum(self.losses) / self.total_samples if self.total_samples > 0 else 0
        return {
            'loss': avg_loss
        }
        
    def log_metrics(self, metrics, prefix=''):
        for key, value in metrics.items():
            print(f"{prefix}{key}: {value:.6f}")
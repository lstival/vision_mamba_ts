"""
Evaluation Package

Contains evaluation utilities for both classification and MAE models:
- Model evaluation functions
- Metrics computation
- Visualization utilities
- Results saving
"""

from .evaluator import ClassificationEvaluator, MAEEvaluator
from .metrics import compute_classification_metrics, compute_mae_metrics
from .visualization import plot_confusion_matrix, plot_training_curves, create_reconstruction_gallery

__all__ = [
    'ClassificationEvaluator',
    'MAEEvaluator', 
    'compute_classification_metrics',
    'compute_mae_metrics',
    'plot_confusion_matrix',
    'plot_training_curves',
    'create_reconstruction_gallery'
]
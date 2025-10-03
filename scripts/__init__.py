"""
Scripts Package

Contains main training and evaluation scripts:
- Training scripts for classification and MAE
- Evaluation scripts  
- Resume training utilities
"""

# Import main functions for CLI entry points
try:
    from .train_vision_mamba import main as train_classifier_main
    from .train_vision_mamba_mae import main as train_mae_main
    from .evaluate_model import main as evaluate_classifier_main
    from .evaluate_mae import main as evaluate_mae_main
    from .resume_training import main as resume_training_main
except ImportError:
    # Handle case where scripts are run directly
    pass

__all__ = [
    'train_classifier_main',
    'train_mae_main', 
    'evaluate_classifier_main',
    'evaluate_mae_main',
    'resume_training_main'
]
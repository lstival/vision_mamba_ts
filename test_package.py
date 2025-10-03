#!/usr/bin/env python3
"""
Test script to verify the package structure and imports work correctly
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all main imports work correctly"""
    print("Testing package imports...")
    
    try:
        # Test main package import
        print("✓ Testing main package import...")
        import vision_mamba
        print(f"  Vision Mamba version: {vision_mamba.__version__}")
        
        # Test model imports
        print("✓ Testing model imports...")
        from vision_mamba.models import (
            VisionMamba, 
            VisionMambaMAE,
            create_vision_mamba_tiny,
            create_vision_mamba_small,
            create_vision_mamba_base
        )
        print("  Models imported successfully")
        
        # Test config imports
        print("✓ Testing config imports...")
        from vision_mamba.config import Config, get_config
        print("  Config imported successfully")
        
        # Test utils imports
        print("✓ Testing utils imports...")
        from vision_mamba.utils import set_seed, get_device
        print("  Utils imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_config_loading():
    """Test that configuration loading works"""
    print("\nTesting configuration loading...")
    
    try:
        from vision_mamba.config import get_config, list_available_configs
        
        # List available configs
        configs = list_available_configs()
        print(f"✓ Found {len(configs)} configuration files:")
        for config in configs:
            print(f"  - {config}")
        
        # Try loading a config
        if configs:
            config = get_config(configs[0])
            print(f"✓ Successfully loaded config: {configs[0]}")
            print(f"  Experiment: {config.experiment_name}")
            print(f"  Dataset: {config.data.dataset_name}")
            print(f"  Model: {config.model.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False


def test_model_creation():
    """Test that model creation works"""
    print("\nTesting model creation...")
    
    try:
        from vision_mamba.models import create_vision_mamba_tiny
        from vision_mamba.config import get_config
        
        # Create a simple model
        model = create_vision_mamba_tiny(
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            dropout=0.1
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Created Vision Mamba Tiny model")
        print(f"  Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("Vision Mamba Package Structure Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_config_loading()
    all_tests_passed &= test_model_creation()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! Package structure is working correctly.")
        print("\nYou can now use the package by:")
        print("1. Installing in development mode: pip install -e .")
        print("2. Running scripts: python scripts/train_vision_mamba.py")
        print("3. Running demos: python demos/demo_training.py")
    else:
        print("❌ Some tests failed. Please check the package structure.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
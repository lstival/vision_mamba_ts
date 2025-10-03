"""
Demo script showing how to use the evaluation scripts
"""
import os
import subprocess
import sys

def run_evaluation():
    """Run evaluation with the best model checkpoint"""
    
    print("Vision Mamba Model Evaluation Demo")
    print("=" * 50)
    
    # Check if best model exists
    best_model_path = "./checkpoints/vision_mamba_cifar10/best_model.pth"
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found at: {best_model_path}")
        print("Please ensure you have trained a model first using:")
        print("  python train_vision_mamba.py --config cifar10")
        return
    
    print(f"✅ Found best model at: {best_model_path}")
    
    # Run enhanced evaluation
    print("\n🚀 Running enhanced evaluation...")
    print("This will:")
    print("  • Load the best CIFAR-10 model")
    print("  • Evaluate on the test set")
    print("  • Generate comprehensive metrics and plots")
    print("  • Save results to evaluation_results/")
    
    try:
        # Method 1: Using the enhanced evaluation script (recommended)
        print("\n📊 Running enhanced evaluation script...")
        cmd = [sys.executable, "enhanced_evaluate.py"]
        subprocess.run(cmd, check=True)
        
        print("\n✅ Enhanced evaluation completed!")
        print("Check the 'evaluation_results' folder for detailed results.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running enhanced evaluation: {e}")
        
        # Fallback: Try the original evaluation script
        print("\n🔄 Trying original evaluation script...")
        try:
            cmd = [sys.executable, "evaluate_model.py", "--checkpoint", best_model_path]
            subprocess.run(cmd, check=True)
            print("✅ Original evaluation completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running original evaluation: {e}")
    
    print("\n📋 Available evaluation options:")
    print("1. Enhanced evaluation (recommended):")
    print("   python enhanced_evaluate.py")
    print("   • Auto-detects best model")
    print("   • Comprehensive metrics and visualizations")
    print("   • Detailed JSON results")
    print()
    print("2. Original evaluation:")
    print("   python evaluate_model.py --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth")
    print("   • Requires manual checkpoint specification")
    print("   • Basic metrics and plots")
    print()
    print("3. With custom options:")
    print("   python enhanced_evaluate.py --experiment vision_mamba_cifar10 --config cifar10 --output_dir custom_results")


if __name__ == "__main__":
    run_evaluation()
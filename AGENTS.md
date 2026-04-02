# AGENTS.md - MRI Motion Classification

## Commands
- **Run AD experiment**: `python MainADExperiment.py` (trains on clean, tests across motion levels)
- **Generate motion data**: `python -m Utils.MotionUtils.GenerateMotion`
- **Run ViT training (PyTorch)**: `python MainViT.py`
- **Run CNN training (Keras)**: `python Main.py`
- **Run all tests**: `python -m pytest tests/`
- **Run single test**: `python -m pytest tests/test_<name>.py::TestClass::test_method -v`

## Architecture
- **MainViT.py**: Entry point for PyTorch Vision Transformer training
- **Main.py**: Legacy entry point for Keras CNN training
- **DeepLearning/**: Neural network models (ViTModel, CNNModel, VAENetwork, CycleGAN)
- **Utils/DataUtils/**: Data loading, dataset creation, augmentation
- **Utils/MotionUtils/**: Synthetic MRI motion generation (GenerateMotion, ImageTransform)
- **Utils/kspace/**: K-space sampling (CartesianSampler)
- **Utils/ActivationMapUtils/**: Model interpretation utilities
- **tests/**: Unit tests using unittest.TestCase

## Dependencies
PyTorch, TensorFlow/Keras (legacy), NumPy, OpenCV (cv2), nibabel (NIfTI loading), pandas, matplotlib, scikit-learn

## Code Style
- Use snake_case for functions/variables, PascalCase for classes
- Tests extend `unittest.TestCase` with `test_` prefix methods
- Imports: standard library, third-party, then local (Utils.*, DeepLearning.*)
- Configure data paths in `Utils/DataUtils/DataGenerator.py` before training
- Model input shape: (256, 256, 1) grayscale images
- Motion classes: M0-M4 (none to severe motion severity)

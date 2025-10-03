"""
Setup script for Vision Mamba package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vision-mamba",
    version="0.1.0",
    author="Vision Mamba Team",
    author_email="",
    description="A hybrid architecture combining State Space Models and Attention for image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vision_mamba_ts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "visualization": [
            "scikit-image>=0.19.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vision-mamba-train=scripts.train_vision_mamba:main",
            "vision-mamba-mae-train=scripts.train_vision_mamba_mae:main",
            "vision-mamba-evaluate=scripts.evaluate_model:main",
            "vision-mamba-mae-evaluate=scripts.evaluate_mae:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vision_mamba": ["configs/*.yaml"],
    },
)
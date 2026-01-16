"""Setup script for GLM-Image training."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="glm-training",
    version="0.1.0",
    description="Training framework for GLM-Image",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GLM Training Contributors",
    url="https://github.com/kat3ri/GLM-training",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "glm-train=train:main",
            "glm-train-ar=train_ar:main",
            "glm-train-dit=train_dit:main",
        ],
    },
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hthta-vit-plus-plus",
    version="1.0.0",
    author="Simer Khurmi, Naincy Yadav, Prisha Sharma, Vidushi Arora, Surbhi Bharti, Ashwini Kumar",
    author_email="simer.live@gmail.com",
    description="HTHTA-ViT++: An Explainable and Efficient Vision Transformer with Hierarchical GRU-Guided Token Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hthta-vit-plus-plus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pre-commit>=2.20.0",
        ],
        "cuda": [
            "nvidia-ml-py3>=7.352.0",
        ],
        "distributed": [
            "accelerate>=0.16.0",
        ],
        "optimization": [
            "torch-optimizer>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hthta-train=src.training.trainer:main",
            "hthta-eval=src.evaluation.evaluator:main",
            "hthta-visualize=src.evaluation.visualizer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [".yaml", ".json", "*.txt"],
    },
    keywords="vision transformer, attention mechanism, explainable AI, deep learning, computer vision",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hthta-vit-plus-plus/issues",
        "Source": "https://github.com/yourusername/hthta-vit-plus-plus",
        "Documentation": "https://hthta-vit-plus-plus.readthedocs.io/",
    },
)

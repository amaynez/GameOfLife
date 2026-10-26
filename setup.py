from setuptools import find_packages, setup


setup(
    name="gameoflife",
    version="0.4.0",
    description="Modernized Conway's Game of Life with advanced backends and adaptive RL jump control",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.8",
        "numpy>=1.26",
        "numba>=0.59",
        "PyYAML>=6.0",
        "imageio>=2.28",
    ],
    extras_require={
        "torch": [
            "torch>=2.3",
            "gymnasium>=0.29",
            "pytorch-lightning>=2.2",
            "tensorboard>=2.16",
        ],
        "rl": [
            "gymnasium>=0.29",
            "pytorch-lightning>=2.2",
            "tensorboard>=2.16",
            "torch>=2.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "gameoflife=gameoflife.cli:main",
            "gameoflife-train-rl=gameoflife.rl.train:main",
            "gameoflife-rl-train=gameoflife.rl.train:main",
            "gameoflife-rl-eval=gameoflife.rl.eval:main",
        ]
    },
    url="https://github.com/Kiy-K/GameOfLife",
    project_urls={
        "Upstream": "https://github.com/amaynez/GameOfLife",
        "Fork": "https://github.com/Kiy-K/GameOfLife",
    },
)

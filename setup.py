import os
import re
import sys

from setuptools import find_packages, setup

deps = [
    "torch>=2.0",
    "safetensors",
    "requests",
    "click",
]

setup(
    name="checkpoint-tools",
    version="0.2.0", # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Useful tools for working with pytorch checkpoints and popular machine learning libraries.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"taproot": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=deps,
    extras_require={
    },
    entry_points={
        "console_scripts": [
            "checkpoint-tools = checkpoint_tools.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, 13)],
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fastmaxent",
    version="0.1.0",
    description="Fast unbiased sampling of networks with given expected degrees and strengths.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xuanchi Li, Xin Wang, Sadamori Kojaku",
    url="https://github.com/EKUL-Skywalker/fastmaxent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.56.0",
    ],
    extras_require={
        "inference": ["nemtropy>=3.0.0", "pandas>=1.3.0"],
        "examples": ["nemtropy>=3.0.0", "pandas>=1.3.0"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "fastmaxent-cli=fastmaxent.cli:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="network sampling, configuration model, maximum entropy, graph theory",
)

from setuptools import setup, find_packages

setup(
    name="fastmaxent",
    version="0.1.0",
    description="Fast unbiased sampling of networks with given expected degrees and strengths.",
    author="Xuanchi Li, Xin Wang, Sadamori Kojaku",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "numba",
        "nemtropy"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
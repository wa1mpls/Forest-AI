from setuptools import setup, find_packages

setup(
    name="forest-ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "scikit-learn>=0.24.2",
        "rasterio>=1.2.10",
        "earthengine-api>=0.1.270",
        "tqdm>=4.62.3",
        "pathlib>=1.0.1"
    ],
) 
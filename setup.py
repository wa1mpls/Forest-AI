from setuptools import setup, find_packages
def read_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

setup(
    name="forest-ai",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),
) 
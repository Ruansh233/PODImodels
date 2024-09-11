from setuptools import setup, find_packages
import setuptools


setup(
    name='PODImodels',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='An example python package',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    author='Shenhui Ruan',
    author_email='shenhui.ruan@kit.edu',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

from setuptools import setup, find_packages

setup(
    name='multiple_annotators_segmentation',
    version='0.1.5.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'matplotlib',
        'numpy',
        'wandb'
    ],
    author='Lucas Iturriago',
    author_email='liturriago@unal.edu.co',
    description='Library for multiple annotators segmentation, using custom Losses and Keras models',
    url='https://github.com/UN-GCPDS/python-gcpds.multiple_annotators_segmentation.git',
    license='LICENSE'
)
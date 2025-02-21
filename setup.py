from setuptools import setup, find_packages

setup(
    name='multiple_annotators_segmentation',
    version='0.1.0',
    packages=find_packages(where='multiple_annotators_segmentation'),
    package_dir={'': 'multiple_annotators_segmentation'},
    install_requires=[
        'tensorflow==2.15.0',
        'matplotlib',
        'numpy',
        'classification_models @ git+https://github.com/qubvel/classification_models.git'
    ],
    author='Lucas Iturriago',
    author_email='liturriago@unal.edu.co',
    description='Library for multiple annotators segmentation, using custom Losses and Keras models',
    url='https://github.com/UN-GCPDS/python-gcpds.multiple_annotators_segmentation.git',
)
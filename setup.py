from setuptools import setup, find_packages

setup(
    name='synthetic_data_generator_package',
    version='0.1',                  
    packages=find_packages(),       
    install_requires=[              
        'numpy',
        'scipy'
    ],
    author='Shwasa',
    url='https://github.com/Shwasa/synthetic-data-generator',
)
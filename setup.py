from setuptools import find_packages, setup

setup(
    name='tensor-dynamic',
    version='0.0.1',
    author='Daniel Slater',
    description='Library for dynamically learning network topology',
    url='https://github.com/DanielSlater/tensordynamic',
    packages=(find_packages(exclude=['tensor_dynamic'])),
    install_requires=[
        'tensorflow>=1.0.1',
    ]
)

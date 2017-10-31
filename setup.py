from setuptools import find_packages, setup

setup(
    name='tensor-dynamic',
    version='0.0.1',
    author='Daniel Slater',
    description='Library for dynamicly learning network topology',
    url='https://github.com/DanielSlater/tensordynamic',
    packages=(find_packages(exclude=['*tests*'])),
    install_requires=[
        'tensorflow>=1.0.1',
    ]
)

from setuptools import setup, find_packages
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = list(f.read().splitlines())
setup(
    name='MUFPO-ML',
    version='0.0.6',
    packages=find_packages(),
    install_requires=install_requires,
    author='Vitvara Varavithya',
    author_email='grid.vitvara@gmail.com',
)
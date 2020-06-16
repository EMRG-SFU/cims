from setuptools import setup, find_packages

setup(
   name='pyCIMS',
   version='0.1',
   description='Python version of the CIMS model',
   author='Jillian Anderson & Maude Lachaine-Loiselle',
   author_email='jilliana@sfu.ca',
   packages=find_packages(),
   install_requires=['networkx', 'numpy', 'pandas'],  # external packages as dependencies
)
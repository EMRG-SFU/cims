from setuptools import setup, find_packages

setup(
   name='CIMS',
   version='0.1',
   description='Python implementation of the CIMS model',
   author='Jillian Anderson & Bradford Griffin',
   author_email='jilliana@sfu.ca',
   packages=find_packages(),
   install_requires=['networkx',
                     'numpy',
                     'pandas>=1.2',
                     'xlrd',
                     'pyxlsb',
                     'scipy',
                     'seaborn>=0.13.2'
                     ],  # external packages as dependencies
)

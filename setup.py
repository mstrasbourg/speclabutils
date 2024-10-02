from setuptools import setup

setup(
   name='speclabutils',
   version='1.0',
   description='A useful module',
   author='MCS, NJB; Users of the Borys and Schuck labs',
   author_email='strasbourg.matt@gmail.com',
   packages=['speclabutils'],  #same as name
   install_requires=['numpy', 'h5py', 'matplotlib'], #external packages as dependencies
)
from setuptools import setup

setup(
   name='mprl',
   version='1.0',
   description='Multiplayer Rienforcement Learning. Research Code Release for P2SRO. https://arxiv.org/abs/2006.08555',
   author='J.B. Lanier, Stephen McAleer',
   author_email='johnblanier@gmail.com',
   packages=['mprl'],  #same as name
   install_requires=['ray==0.8', 'numpy', 'progress', 'pyglet', 'gym', 'pandas', 'tabulate', 'bidict', 'minio', 'plotly', 'opencv-python==4.1.0.25', 'lz4'], #external packages as dependencies
)
from setuptools import setup

setup(
   name='pomprl',
   version='1.0',
   description='Partially Observable Multiplayer Rienforcement Learning. Research Code Release for P2SRO. https://arxiv.org/abs/2006.08555',
   author='J.B. Lanier, Stephen McAleer',
   author_email='johnblanier@gmail.com',
   packages=['pomprl'],  #same as name
   install_requires=['numpy', 'progress', 'pyglet', 'gym'], #external packages as dependencies
)
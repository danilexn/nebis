from setuptools import setup, find_packages

setup(
  name = 'nebis',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'nebis - Neural Networks for Biological Set Representations',
  author = 'Daniel Leon Perinan',
  author_email = 'daniel.leon-perinan@mailbox.tu-dresden.de',
  url = 'https://github.com/danilexn/nebis',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'bioinformatics',
    'set representation',
    'sequences'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
    'torchvision'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
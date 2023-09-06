#!/usr/bin/env python

from setuptools import setup

setup(name='pyvsa',
      version='0.1',
      description='Vector Symbolic Architecture',
      author='Allen Pan',
      author_email='allpan@email.com',
      url='',
      packages=['vsa'],
      install_requires=[
        "torch>=1.9.0"
    ],
)

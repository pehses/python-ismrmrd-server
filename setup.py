# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:49:16 2017

@author: Marten Veldmann
"""

from setuptools import setup, find_packages


setup(name='ismrmrd_client',
      version="1.0",
      description="Client for streaming ISMRMRD data",
      scripts=['client.py','connection.py','constants.py'],
      packages=find_packages(exclude=("docker","parameter_maps")),
      install_requires=[
          'numpy',
          'ismrmrd>=1.9.5',
          'h5py'
      ],
      zip_safe=False,
      )
      
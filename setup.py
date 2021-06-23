# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:49:16 2017

@author: Marten Veldmann
"""

from setuptools import setup


setup(name='ismrmrd_client',
      version="1.0",
      description="Client for streaming ISMRMRD data",
      scripts=['client.py','connection.py','constants.py'],
      zip_safe=False,
      )
      
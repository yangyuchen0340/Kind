#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:49:30 2019

@author: yangyc
"""

from setuptools import setup


setup(
    name='Kind',
    version='0.2.0',
    description="Toolbox for data clustering based on K-indicators model",
    url='https://github.com/yangyuchen0340/Kind',
    author='Yuchen Yang, Feiyu Chen and Yin Zhang',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='clustering, machine learning,numpy,scipy,sklearn, optimization',
    packages=['Kind'],
    install_requires=['numpy>=1.15', 'scipy>=0.17', 'scikit-learn>=0.19', 'six>=1.10'],
)

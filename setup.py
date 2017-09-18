"""
Setup script for the Classifier package.

All following steps asume that you have a console/terminal open in the directory of this file.

Run this script to install the package using:
    pip install .

To install the package for local development use:
    pip install -e .

To run all the tests for the classifiers use:
    python setup.py test

"""

from setuptools import setup
if __name__ == "__main__":
    setup(name='Classifiers',
          version='0.0.0',
          description='Mashine Learning Classifiers writen in plain python.',
          url='',
          author='Benjamin Bein',
          author_email='',
          license='MIT',
          test_suite='nose.collector',
          tests_require=['nose'],
          packages=['Classifiers'],
          install_requires=[],
          zip_safe=False)

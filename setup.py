from __future__ import absolute_import, print_function

import io
import re
from glob import glob
from os.path import basename, dirname, join, splitext

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='thesis-explorations',
    version='0.1.0',
    license='BSD-3 Clause',
    description='Collection of various experiments and explorations for my MSc thesis project.',
    long_description="",
    author='Max Peeperkorn',
    author_email='post@maxpeeperkorn.nl',
    url='https://github.com/maxvstheuniverse/thesis-explorations',
    packages=find_packages('src'),
    package_dir={
        '': 'src'
    },
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    project_urls={
        'Changelog': '',
        'Issue Tracker': '',
    },
    python_requires='>=3.7',
    install_requires=[],
    extras_require={},
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={
        'console_scripts': [
            'bng=basic_naming_game.__main__:main',
            'ang=ae_naming_game.__main__:main'
        ]
    },
)

import os
import re
import sys
from setuptools import setup


install_requires = ['torch==1.1.0.post2']


if sys.version_info < (3, 6, 0):
    raise RuntimeError('torch-inspect requires Python 3.6.0+')


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'torch_inspect', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            raise RuntimeError(
                'Cannot find version in torch_inspect/__init__.py'
            )


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: OS Independent',
    'Development Status :: 4 - Beta',
]


setup(
    name='torch-inspect',
    version=read_version(),
    description=(''),
    long_description='\n\n'.join((read('README.rst'), read('CHANGES.rst'))),
    classifiers=classifiers,
    platforms=['POSIX'],
    author='Nikolay Novik',
    author_email='nickolainovik@gmail.com',
    url='',
    download_url='https://pypi.python.org/pypi/torch-inspect',
    license='Apache 2',
    packages=['torch_inspect'],
    install_requires=install_requires,
    keywords=['torch-inspect', 'pytorch'],
    zip_safe=True,
    include_package_data=True,
)

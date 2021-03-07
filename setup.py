import os.path
import sys

import setuptools

from mlit import __description__, __version__

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)


with open('README.md', 'r') as fh:
    long_description = fh.read()


setup_args = dict(
    name='mlit',
    version=__version__,
    # url=None,
    author='roman@kalyakin.com',
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers[torch]>=4.3.3',
        'onnxruntime>=1.7.0'
    ],
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.7",
    # license=None,
    platforms="Linux, Mac OS X, Windows",
    keywords=["ML", "hugginface", "onnx"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)

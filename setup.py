from setuptools import setup, find_packages
import re
import io
import pathlib

__package_name__ = 'goodness_of_fit'

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open(__package_name__ + '/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name=__package_name__,
    version=__version__,
    description='Function set for goodness of fit measure between two signals',
    long_description=README,
    author='Simon Delmas',
    author_email='delmas.simon@gmail.com',
    url='https://github.com/SimonDelmas/goodness_of_fit',
    license="GLP-2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Framework :: Pytest",
        "Intended Audience :: Science/Research"
    ],
    packages=find_packages(),
    install_requires=['numpy']
)
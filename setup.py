from setuptools import setup, find_packages
import re
import io


__package_name__ = 'goodness_of_fit'

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open(__package_name__ + '/__init__.py', encoding='utf_8_sig').read()
    ).group(1)


setup(
    name=__package_name__,
    version=__version__,
    packages=find_packages(),
    url='',
    license='',
    author='Simon Delmas',
    author_email='delmas.simon@gmail.com',
    description='',
    install_requires=['numpy']
)
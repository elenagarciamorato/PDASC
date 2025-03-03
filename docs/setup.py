try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

requires = []


with open('requirements.txt') as fp:
    requires = fp.read()

setup(
    name='PDASC',
    version='0.0.1',
    url='https://github.com/elenagarciamorato/PDASC',
    license='',
    author='Elena Garcia-Morato, Maria Jesus Algar, Cesar Alfaro, Felipe Ortega, Javier Gomez and Javier M. Moguerza',
    author_email='elena.garciamorato@urjc.es, mariajesus.algar@urjc.es, cesar.alfaro@urjc.es, felipe.ortega@urjc.es, javier.gomez@urjc.es and javier.moguerza@urjc.es',
    description='',
    packages=find_packages(),
    platforms='any',
    install_requires=requires
)

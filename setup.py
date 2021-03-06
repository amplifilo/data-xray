from setuptools import setup, find_packages

#from distutils.core import setup


setup(
    name='data_xray',
    version='0.5.3',
    author='Petro Maksymovych',
    author_email='pmax20@gmail.com',
    maintainer='Petro Maksymovych',
    maintainer_email='pmax20@gmail.com',
    packages=find_packages(),
    home_page=['https://github.com/amplifilo/data-xray'],
    license='LICENSE.txt',
    description='Pythonic cure for the hyperspectral morass',
    long_description=open('README.rst').read(),
    install_requires=['tqdm','python-pptx','pyperclip','matplotlib-scalebar','yattag','deepdish','xarray'],
    extras_require = { 'docs': [""],},
    test_suite = "",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3.6'
    ],
)

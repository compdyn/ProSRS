"""
Copyright (C) 2019 Chenchao Shou

Licensed under Illinois Open Source License (see the file LICENSE). For more information
about the license, see http://otm.illinois.edu/disclose-protect/illinois-open-source-license.

"""
import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="prosrs",
    version="0.0.1",
    description="A tree-based parallel surrogate optimization algorithm for optimizing noisy expensive functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/compdyn/ProSRS",
    author="Chenchao Shou",
    author_email="cshou3@illinois.edu",
    license="University of Illinois/NCSA Open Source",
    classifiers=[
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Natural Language :: English",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "License :: OSI Approved :: University of Illinois/NCSA Open Source License",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development"
    ],
    keywords="optimization algorithm",
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=['numpy', 'scipy', 'matplotlib', 'pyDOE', 'pathos', 'scikit-learn']      
)
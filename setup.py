import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "jester",
    version = "0.0.1",
    author = "Brielin Brown",
    author_email = "brielin.brown@gmail.com",
    description = ("A tool for computing genome-wide joint association tests"
                   "and determining the multiple testing correction."),
    license = "GNUV2",
    keywords = "GWAS, joint test, disease association",
    url = "https://github.com/brielin/Jester",
    packages=['jester', 'jester/tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    entry_points={
        'console_scripts': ['jester = jester.__main__:main'],
    },
)

# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "Core.processing_script",
        ["Core/processing_script.py"],
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()]
)
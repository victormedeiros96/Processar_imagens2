# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define o módulo a ser compilado
ext_modules = [
    Extension(
        "Core.ai_processing_script",
        ["Core/ai_processing_script.py"],
        # Adicione outras dependências de compilação se necessário
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()] # Essencial para compilar código que usa numpy
)
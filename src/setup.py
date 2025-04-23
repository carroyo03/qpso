import pybind11
from setuptools import setup, Extension #type: ignore
from setuptools.command.build_ext import build_ext #type: ignore
import sys
import os

class BuildExt(build_ext):
    
    def build_extensions(self):
        
        if sys.platform == "darwin":
            
            for ext in self.extensions:
                ext.extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                ext.extra_link_args += ["-lomp"]
                
                if os.path.exists('/opt/homebrew/opt/libomp'):
                    ext.include_dirs.append('/opt/homebrew/opt/libomp/include')
                    ext.library_dirs.append('/opt/homebrew/opt/libomp/lib')

        else:
            for ext in self.extensions:
                ext.extra_compile_args += ["-fopenmp"]
                ext.extra_link_args += ["-fopenmp"]

        
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'cpp.pso_core',
        sources=['cpp/pso_core.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11'],
    )
]

setup(
    name="qpso",
    version="0.1.0",
    description="Particle Swarm Optimization with C++/OpenMP acceleration",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
# This file is part of crosci, licensed under Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA).
# See LICENSE.txt for more details.

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import platform
os_name = platform.system()

if os_name=="Windows":
    extra_compile_args = ['/openmp']
    extra_link_args = []
elif os_name=="Darwin":
    # extra_compile_args = ['-fopenmp=libomp'] 
    # extra_link_args = []
    extra_compile_args = ['-Xclang', '-fopenmp', '-isysroot', os.getenv('SDKROOT', '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'), '-mmacosx-version-min=14.0'] # added by Miri for Mac ARM64
    extra_link_args = ['-lomp', '-isysroot', os.getenv('SDKROOT', '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'), '-mmacosx-version-min=14.0'] # added by Miri for Mac ARM64
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

extensions = [Extension('crosci.compiled.run_DFA',
                        [os.path.join(dir_path,'c/run_DFA.pyx'), os.path.join(dir_path,'c/dfa.c')],
                         extra_compile_args = extra_compile_args,
                         extra_link_args = extra_link_args,
                         include_dirs=[os.getenv('CPPFLAGS', '/opt/homebrew/opt/llvm/include')] # added by Miri for Mac ARM64
),
              Extension('crosci.compiled.run_fEI',
                        [os.path.join(dir_path,'c/run_fEI.pyx'), os.path.join(dir_path,'c/fEI.c')],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args,
                        include_dirs=[os.getenv('CPPFLAGS', '/opt/homebrew/opt/llvm/include')] # added by Miri for Mac ARM64
),]

setup(
    ext_modules = cythonize(extensions)
)
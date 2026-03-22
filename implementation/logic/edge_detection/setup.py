from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sysconfig
import os

# Get Python’s library name and directory
python_lib = sysconfig.get_config_var("LIBRARY")  # e.g. python313.lib
python_lib_dir = sysconfig.get_config_var("LIBDIR")  # e.g. C:\Python313\libs

local_src = "src"
external_module_root = os.path.join("..", "additional_modules")

ext_modules = [
    Extension(
        "EdgeDetector",
        sources=[
            "bindings.cpp",
            os.path.join(local_src, "additional_functions.cpp"),
            os.path.join(local_src, "edge_detector.cpp"),
            os.path.join(external_module_root, "src", "matrix_convert.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            "include",
            os.path.join(external_module_root, "include"),
        ],
        library_dirs=[python_lib_dir],
        libraries=["python313"],  # e.g. 'python313'
        language="c++",
        extra_compile_args=["/std:c++20"],  # For MSVC
    )
]

setup(
    name="EdgeDetector",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

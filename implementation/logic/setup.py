from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sysconfig
import os

# Get Python’s library name and directory
python_lib = sysconfig.get_config_var("LIBRARY")  # e.g. python313.lib
python_lib_dir = sysconfig.get_config_var("LIBDIR")  # e.g. C:\Python313\libs

print(f"python_lib: {python_lib}")
print(f"python_lib_dir: {python_lib_dir}")

ext_modules = [
    Extension(
        "geometry",
        sources=[
            "bindings.cpp",
            os.path.join("src", "additional_functions.cpp"),
            os.path.join("src", "process_image.cpp"),
            os.path.join("src", "edge_detector.cpp"),
            os.path.join("src", "matrix_converter.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            "include",
            "src",
        ],
        library_dirs=[python_lib_dir],
        libraries=["python313"],  # e.g. 'python313'
        language="c++",
        extra_compile_args=["/std:c++20"],  # For MSVC
    )
]

setup(
    name="geometry",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

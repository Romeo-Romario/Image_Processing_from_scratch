# General instructions

### To build a package with c++

1. Open Developer command prompt
2. Type in `set CL=/EHsc /std:c++20` to set correct enviroment values
3. Navigate to your project directory
4. Use command `Cl program.cpp src\additional_functions.cpp` -> The **result** will be `{first_file}`.exe

### To build a module that will be used in python code

1. Just open a terminal in VS code
2. Navigate to folder with file **setup.py**
3. run the command `python setup.py build_ext --inplace`

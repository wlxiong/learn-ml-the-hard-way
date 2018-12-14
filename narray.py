import numpy

cupy_imported = False
try:
    import cupy
    z = cupy.zeros((1,1))
    cupy_imported = True
except:
    pass


lib = numpy

def set_library(lib_name):
    global lib
    lib_name = lib_name.lower()
    if lib_name == 'numpy':
        lib = numpy
    elif lib_name == 'cupy':
        if not cupy_imported:
            raise ImportError("Cannot import cupy")
        lib = cupy
    return lib

def get_library():
    return lib

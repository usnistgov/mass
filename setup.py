import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

# See https://stackoverflow.com/a/60751886/3251234 for how this could be configured
# to install numpy and/or Cython on a system that lacks them. One provides a
# cmdclass={"build": build}, to setup, where `build` is a subclass that imports
# these packages only after they have been installed.

if __name__ == "__main__":
    def ext(name):
        path = name.replace(".", "/")
        sources = [path + ".pyx"]
        return Extension(name, sources=sources, include_dirs=[np.get_include()])

    cython_extensions = [
        "mass.core.analysis_algorithms",
    ]

    extensions = []
    package_data = {}

    for name in cython_extensions:
        module, fname = name.rsplit(".", 1)
        extensions.append(ext(name))

        sources = package_data.get(module, [])
        sources.append(fname)
        package_data[module] = sources

    setup(
        ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
        package_data=package_data,
    )

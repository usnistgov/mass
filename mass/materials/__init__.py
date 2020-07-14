try:
    import xraylib

except ImportError:
    print("WARNING: Could not import mass.materials package. It requires xraylib.")
    print("See https://github.com/tschoonj/xraylib/wiki for installation instructions.\n")

if "xraylib" in locals():
    import mass.materials.efficiency_models
    from .efficiency_models import *

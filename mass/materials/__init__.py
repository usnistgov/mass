try:
    import xraylib

    import mass.materials.efficiency_models

    from .efficiency_models import *

except ModuleNotFoundError as e:
    print('** Skipping module mass.materials, because it requires the "xraylib" python package.')
    print('** Please see https://github.com/tschoonj/xraylib/wiki for installation instructions.')

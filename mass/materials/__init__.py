try:
    import xraydb
    import mass.materials.efficiency_models
    from .efficiency_models import *

except ModuleNotFoundError:
    print('** Skipping module mass.materials, because it requires the "xraydb" python package.')
    print('** Please see https://xraypy.github.io/XrayDB/installation.html for installation instructions.')

try:
    import xraydb
    from . import efficiency_models
    from .efficiency_models import FilterStack, Film, AlFilmWithOxide, AlFilmWithPolymer, \
        LEX_HT, filterstack_models

except ModuleNotFoundError:
    print('** Skipping module mass.materials, because it requires the "xraydb" python package.')
    print('** Please see https://xraypy.github.io/XrayDB/installation.html for installation instructions.')

__all__ = ["xraydb", "efficiency_models", "FilterStack", "Film",
           "AlFilmWithOxide", "AlFilmWithPolymer", "LEX_HT", "filterstack_models"]

from mass.xray_filters import *

# EBIT Instrument
EBIT_filter_stack = FilterStack(name='EBIT Filter Stack')
EBIT_filter_stack.add(Film(name='Electroplated Au Absorber', material='Au', thickness_nm=965.5, absorber=True))
EBIT_filter_stack.add(Film(name='50mK Filter', material='Al', thickness_nm=112.5))
EBIT_filter_stack.add(Film(name='3K Filter', material='Al', thickness_nm=108.5))
filter_50K = FilterStack(name='50K Filter')
filter_50K.add(Film(name='Al Film', material='Al', thickness_nm=102.6))
filter_50K.add(Mesh(name='Ni Mesh', material='Ni', thickness_nm=15.0e3, fill_fraction=0.17))
EBIT_filter_stack.add(filter_50K)
EBIT_filter_stack.add(LEX_HT('Luxel Window TES'))
EBIT_filter_stack.add(LEX_HT('Luxel Window EBIT'))

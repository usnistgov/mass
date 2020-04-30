import mass.xray_filters as xray_filters


# Name filter stack level object, create associated dict
EBIT_filter_stack_name = 'EBIT Filter Stack'
EBIT_filter_dict = {}
# Name filter level objects, describe filter components
# Units in keV for energy, cgs for everything else
EBIT_filter_dict['Absorber'] = {}
EBIT_filter_dict['Absorber']['Evaporated Au'] = {'component_type': 'AbsorberFromThickness', 'material': 'Au', 'thickness': 965.5e-7}

EBIT_filter_dict['Filter 50mK'] = {}
EBIT_filter_dict['Filter 50mK']['Al Film'] = {'component_type': 'FilmFromThickness', 'material': 'Al', 'thickness': 112.5e-7}

EBIT_filter_dict['Filter 3K'] = {}
EBIT_filter_dict['Filter 3K']['Al Film'] = {'component_type': 'FilmFromThickness', 'material': 'Al', 'thickness': 108.5e-7}

EBIT_filter_dict['Filter 50K'] = {}
EBIT_filter_dict['Filter 50K']['Al Film'] = {'component_type': 'FilmFromThickness', 'material': 'Al', 'thickness': 102.6e-7}
EBIT_filter_dict['Filter 50K']['Ni Mesh'] = {'component_type': 'MeshFromThickness', 'material': 'Ni', 'thickness': 15.0e-4, 'fraction_blocked': 0.17}

EBIT_filter_dict['Luxel Window TES'] = {'component_type': 'LuxelVacuumWindow'}

EBIT_filter_dict['Luxel Window EBIT'] = {'component_type': 'LuxelVacuumWindow'}

# Create filter stack object with given name and import_dict
EBIT_filter_stack = xray_filters.FilterObject(name=EBIT_filter_stack_name, import_dict=EBIT_filter_dict)




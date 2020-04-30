import xraylib
import numpy as np
import pylab as plt

class FilterObject():
    def __init__(self, name, import_dict=None):
        self.name = name
        self.components = {}
        if import_dict is not None:
            self.import_dict = import_dict
            self.setup_from_dict(self.import_dict)

    def setup_from_dict(self, import_dict):
        for component_name in list(import_dict.keys()):
            component_dict = import_dict[component_name]
            if 'component_type' in component_dict:
                component_type = component_dict['component_type']
            else:
                component_type = 'FilterObject'
            if type(component_dict) == dict:
                self.add_component(component_name=component_name, component_type=component_type, component_dict=component_dict)
            
    def add_component(self, component_name, component_type, component_dict):
        valid_component_types = [None, 'FilterObject', 'FilmFromThickness', 'MeshFromThickness', 'AbsorberFromThickness',
        'FilmFromAreaDensity', 'MeshFromAreaDensity', 'LuxelVacuumWindow']
        assert component_type in valid_component_types, \
            '{} is not a valid component type, must be one of {}'.format(component_type, valid_component_types)
        if np.logical_or(component_type == None, component_type=='FilterObject'):
            self.components[component_name] = FilterObject(name=component_name, import_dict=component_dict)

        elif np.logical_or(component_type == 'FilmFromThickness', component_type == 'AbsorberFromThickness'):
            assert 'material' in component_dict, 'material parameter required for component_type FilmFromThickness, AbsorberFromThickness'
            material = component_dict['material']
            assert 'thickness' in component_dict, 'thickness parameter required for component_type FilmFromThickness, AbsorberFromThickness'
            thickness = component_dict['thickness']
            if 'density' in component_dict:
                density = component_dict['density']
            else:
                density = None
            if component_type == 'FilmFromThickness':
                self.components[component_name] = FilmFromThickness(component_name=component_name, material=material, 
                thickness=thickness, density=density)
            else:
                self.components[component_name] = AbsorberFromThickness(component_name=component_name, material=material, 
                thickness=thickness, density=density)
        
        elif component_type == 'FilmFromAreaDensity':
            assert 'material' in component_dict, 'material parameter required for component_type FilmFromAreaDensity'
            material = component_dict['material']
            assert 'area_density' in component_dict, 'area_density parameter required for component_type FilmFromAreaDensity'
            area_density = component_dict['area_density']
            self.components[component_name] = FilmFromAreaDensity(component_name=component_name, material=material, area_density=area_density)

        elif component_type == 'MeshFromThickness':
            assert 'material' in component_dict, 'material parameter required for component_type FilmFromThickness'
            material = component_dict['material']
            assert 'thickness' in component_dict, 'thickness parameter required for component_type FilmFromThickness'
            thickness = component_dict['thickness']
            assert 'fraction_blocked' in component_dict, 'fraction_blocked parameter required for component_type FilmFromThickness'
            fraction_blocked = component_dict['fraction_blocked']
            if 'density' in component_dict:
                density = component_dict['density']
            else:
                density = None
            self.components[component_name] = MeshFromThickness(component_name=component_name, material=material, 
            thickness=thickness, fraction_blocked=fraction_blocked, density=density)

        elif component_type == 'MeshFromAreaDensity':
            assert 'material' in component_dict, 'material parameter required for component_type MeshFromAreaDensity'
            material = component_dict['material']
            assert 'area_density' in component_dict, 'area_density parameter required for component_type MeshFromAreaDensity'
            area_density = component_dict['area_density']
            assert 'fraction_blocked' in component_dict, 'fraction_blocked parameter required for component_type MeshFromAreaDensity'
            fraction_blocked = component_dict['fraction_blocked']
            self.components[component_name] = MeshFromAreaDensity(component_name=component_name, material=material, 
            area_density=area_density, fraction_blocked=fraction_blocked)

        elif component_type == 'LuxelVacuumWindow':
            self.components[component_name] = LuxelVacuumWindow(component_name=component_name)

    def get_efficiency(self, xray_energies):
        individual_efficiency = np.array([iComponent.get_efficiency(xray_energies) for iComponent in list(self.components.values())])
        efficiency = np.prod(individual_efficiency, axis=0)
        return efficiency

    def plot_efficiency(self, xray_energies, ax=None):
        efficiency = self.get_efficiency(xray_energies)
        if ax==None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xray_energies, efficiency*100.0)
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel('Efficiency (%)')
            ax.set_title('{} Efficiency'.format(self.name))
        else:
            ax.plot(xray_energies, efficiency)

    def plot_component_efficiencies(self, xray_energies):
        assert self.components != {}, '{} has no components to plot'.format(self.name)
        for iComponent in list(self.components.values()):
            iComponent.plot_efficiency(xray_energies)

class FilmFromThickness(FilterObject):
    def __init__(self, component_name, material, thickness, density=None):
        super().__init__(component_name)
        self.material=np.array(material, ndmin=1)
        self.atomic_number = np.array([xraylib.SymbolToAtomicNumber(iMaterial) for iMaterial in self.material], ndmin=1)
        self.num_materials = len(self.material)
        self.thickness=np.array(thickness, ndmin=1)
        if density is not None:
            self.density = np.array(density, ndmin=1)
        else:
            self.density = np.array([xraylib.ElementDensity(iAtomicNumber) for iAtomicNumber in self.atomic_number], ndmin=1)

    def get_efficiency(self, xray_energies):
        linear_attenuation = np.array([np.array([xraylib.CS_Total_CP(self.material[iMaterialIndex], iEnergy) for iEnergy in xray_energies]) \
            * iDensity for iMaterialIndex, iDensity in enumerate(self.density)])
        optical_depth = linear_attenuation * self.thickness.reshape(self.num_materials,1)
        individual_transmittance = np.exp(-optical_depth)
        transmittance = np.prod(individual_transmittance, axis=0)
        return transmittance

class AbsorberFromThickness(FilmFromThickness):
    def __init__(self, component_name, material, thickness, density=None):
        super().__init__(component_name, material, thickness, density)

    def get_efficiency(self, xray_energies):
        transmittance = super().get_efficiency(xray_energies)
        absorption = 1.0 - transmittance
        return absorption


class FilmFromAreaDensity(FilterObject):
    def __init__(self, component_name, material, area_density):
        super().__init__(component_name)
        self.material=np.array(material, ndmin=1)
        self.num_materials = len(self.material)
        self.area_density=np.array(area_density, ndmin=1)

    def get_efficiency(self, xray_energies):
        optical_depth = np.array([[xraylib.CS_Total_CP(iMaterial, iEnergy) * self.area_density[iMaterialIndex] \
            for iEnergy in xray_energies] for iMaterialIndex, iMaterial in enumerate(self.material)])
        individual_transmittance = np.exp(-optical_depth)
        transmittance = np.prod(individual_transmittance, axis=0)
        return transmittance


class MeshFromThickness(FilterObject):
    def __init__(self, component_name, material, thickness, fraction_blocked, density=None):
        super().__init__(component_name)
        self.material=np.array(material, ndmin=1)
        self.atomic_number = np.array([xraylib.SymbolToAtomicNumber(iMaterial) for iMaterial in self.material], ndmin=1)
        self.num_materials = len(self.material)
        self.thickness=np.array(thickness, ndmin=1)
        self.fraction_blocked = fraction_blocked
        if density is not None:
            self.density = np.array(density, ndmin=1)
        else:
            self.density = np.array([xraylib.ElementDensity(iAtomicNumber) for iAtomicNumber in self.atomic_number], ndmin=1)

    def get_efficiency(self, xray_energies):
        linear_attenuation = np.array([np.array([xraylib.CS_Total_CP(self.material[iMaterialIndex], iEnergy) for iEnergy in xray_energies]) \
            * iDensity for iMaterialIndex, iDensity in enumerate(self.density)])
        optical_depth = linear_attenuation * self.thickness.reshape(self.num_materials,1)
        individual_transmittance = np.exp(-optical_depth)
        transmittance_unscaled = np.prod(individual_transmittance, axis=0)
        transmittance = (1-self.fraction_blocked) + transmittance_unscaled*self.fraction_blocked
        return transmittance

class MeshFromAreaDensity(FilterObject):
    def __init__(self, component_name, material, area_density, fraction_blocked):
        super().__init__(component_name)
        self.material=np.array(material, ndmin=1)
        self.num_materials = len(self.material)
        self.area_density=np.array(area_density, ndmin=1)
        self.fraction_blocked = fraction_blocked

    def get_efficiency(self, xray_energies):
        optical_depth = np.array([[xraylib.CS_Total_CP(iMaterial, iEnergy) * self.area_density[iMaterialIndex] \
            for iEnergy in xray_energies] for iMaterialIndex, iMaterial in enumerate(self.material)])
        individual_transmittance = np.exp(-optical_depth)
        transmittance_unscaled = np.prod(individual_transmittance, axis=0)
        transmittance = (1-self.fraction_blocked) + transmittance_unscaled*self.fraction_blocked
        return transmittance

class LuxelVacuumWindow(FilterObject):
    def __init__(self, component_name, film_material=None, film_area_density=None, mesh_material=None, 
    mesh_area_density=None, mesh_fraction_blocked=None):
        super().__init__(component_name)
        film_dict = {}
        # Some typical Al + polyimide measurements from Luxel
        if film_material is None:
            film_material = ['C', 'H', 'N', 'O', 'Al']
        film_dict['material'] = film_material
        if film_area_density is None:
            film_area_density = [6.7e-5, 2.6e-6, 7.2e-6, 1.7e-5, 1.7e-5]
        film_dict['area_density'] = film_area_density
        film_name = 'Luxel Window Film'
        self.add_component(component_name=film_name, component_type='FilmFromAreaDensity', component_dict=film_dict)
        mesh_dict = {}
        # AISI type 304 stainless steel, typical values
        if mesh_material is None:
            mesh_material = ['Fe','Cr', 'Ni', 'Mn', 'Si']
        mesh_dict['material'] = mesh_material
        if mesh_area_density is None:
            mesh_thickness = 100.0e-4 # cm
            mesh_density = 8.0 # g/cm^3
            mesh_material_fractions = np.array([0.705, 0.19, 0.09, 0.01, 0.005]) # fraction by weight
            mesh_area_density = mesh_material_fractions * mesh_density * mesh_thickness # g/cm^2
        mesh_dict['area_density'] = mesh_area_density
        if mesh_fraction_blocked is None:
            mesh_fraction_blocked = 0.19
        mesh_dict['fraction_blocked'] = mesh_fraction_blocked
        mesh_name = 'Luxel Window Mesh'
        self.add_component(component_name=mesh_name, component_type='MeshFromAreaDensity', component_dict=mesh_dict)
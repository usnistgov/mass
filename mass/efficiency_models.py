import xraylib
import numpy as np
import pylab as plt

class FilterStack():
    def __init__(self, name):
        self.name = name
        self.components = []

    def add(self, c):
        assert isinstance(c, FilterStack)
        self.components.append(c)

    def add_Film(self, name, material, area_density_g_per_cm2=None, thickness_nm=None, density_g_per_cm3=None, absorber=False):
        c = Film(name=name, material=material, area_density_g_per_cm2=area_density_g_per_cm2, 
        thickness_nm=thickness_nm, density_g_per_cm3=density_g_per_cm3, absorber=absorber)
        self.components.append(c)

    def add_Mesh(self, name, material, area_density_g_per_cm2=None, thickness_nm=None, density_g_per_cm3=None, fill_fraction=None, absorber=False):
        c = Mesh(name=name, material=material, area_density_g_per_cm2=area_density_g_per_cm2, 
        thickness_nm=thickness_nm, density_g_per_cm3=density_g_per_cm3, fill_fraction=fill_fraction, absorber=absorber)
        self.components.append(c)

    def add_AlFilmWithOxide(self, name, thickness_nm, Al_density_g_per_cm3=None, num_oxidized_surfaces=2, oxide_density_g_per_cm3=None):
        c = AlFilmWithOxide(name=name, thickness_nm=thickness_nm, Al_density_g_per_cm3=Al_density_g_per_cm3,
        num_oxidized_surfaces=num_oxidized_surfaces, oxide_density_g_per_cm3=oxide_density_g_per_cm3)
        self.components.append(c)

    def add_AlFilmWithPolymer(self, name, Al_thickness_nm, polymer_thickness_nm, polymer_fractions=None, polymer_density_g_per_cm3=None,
    num_oxidized_surfaces=1, oxide_density_g_per_cm3=None):
        c = AlFilmWithPolymer(name=name, Al_thickness_nm=Al_thickness_nm, polymer_thickness_nm = polymer_thickness_nm, 
        polymer_fractions=polymer_fractions, polymer_density_g_per_cm3=polymer_density_g_per_cm3,
        num_oxidized_surfaces=num_oxidized_surfaces, oxide_density_g_per_cm3=oxide_density_g_per_cm3)
        self.components.append(c)

    def add_LEX_HT(self, name):
        c = LEX_HT(name=name)
        self.components.append(c)

    def get_efficiency(self, xray_energies):
        assert self.components != [], '{} has no components of which to calculate efficiency'.format(self.name)
        individual_efficiency = np.array([iComponent.get_efficiency(xray_energies) for iComponent in self.components])
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
        assert self.components != [], '{} has no components to plot'.format(self.name)
        for iComponent in self.components:
            iComponent.plot_efficiency(xray_energies)

class Film(FilterStack):
    def __init__(self, name, material, area_density_g_per_cm2=None, thickness_nm=None, density_g_per_cm3=None, absorber=False):
        super().__init__(name)
        self.material=np.array(material, ndmin=1)
        self.atomic_number = np.array([xraylib.SymbolToAtomicNumber(iMaterial) for iMaterial in self.material], ndmin=1)
        self.absorber=absorber
        assert np.logical_xor(area_density_g_per_cm2 is not None, thickness_nm is not None), 'must use either area density or thickness'
        assert ~np.logical_and(area_density_g_per_cm2 is not None, density_g_per_cm3 is not None), 'overconstrained, must choose area density or density'
        if area_density_g_per_cm2 is not None:
            self.area_density_g_per_cm2 = np.array(area_density_g_per_cm2, ndmin=1)
        elif thickness_nm is not None:
            self.thickness_nm=np.array(thickness_nm, ndmin=1)
            if density_g_per_cm3 is not None:
                self.density_g_per_cm3 = np.array(density_g_per_cm3, ndmin=1)
            else:
                self.density_g_per_cm3 = np.array([xraylib.ElementDensity(iAtomicNumber) for iAtomicNumber in self.atomic_number], ndmin=1)
            self.area_density_g_per_cm2 = self.density_g_per_cm3 * self.thickness_nm * 1e-7

    def get_efficiency(self, xray_energies_eV):
        num_materials=len(self.material)
        optical_depth = np.array([[xraylib.CS_Total_CP(iMaterial, iEnergy) * self.area_density_g_per_cm2[iMaterialIndex] \
            for iEnergy in xray_energies_eV/1000.0] for iMaterialIndex, iMaterial in enumerate(self.material)])
        individual_transmittance = np.exp(-optical_depth)
        transmittance = np.prod(individual_transmittance, axis=0)
        if self.absorber:
            return 1.0 - transmittance
        else:
            return transmittance

class Mesh(Film):
    def __init__(self, name, material, area_density_g_per_cm2=None, thickness_nm=None, density_g_per_cm3=None, fill_fraction=None, absorber=False):
        assert fill_fraction is not None, 'Mesh requires fill_fraction argument'
        self.fill_fraction = fill_fraction
        super().__init__(name=name, material=material, area_density_g_per_cm2=area_density_g_per_cm2,
        thickness_nm=thickness_nm, density_g_per_cm3=density_g_per_cm3, absorber=absorber)

    def get_efficiency(self, xray_energies_eV):
        mesh_efficiency = super().get_efficiency(xray_energies_eV)
        if self.absorber:
            return mesh_efficiency * self.fill_fraction
        else:
            return (mesh_efficiency * self.fill_fraction) + (1.0 - self.fill_fraction)

class AlFilmWithOxide(Film):
    def __init__(self, name, thickness_nm, Al_density_g_per_cm3=None, num_oxidized_surfaces=2, oxide_density_g_per_cm3=None):
        assert num_oxidized_surfaces in [1,2], 'only 1 or 2 oxidzed surfaces allowed'
        if Al_density_g_per_cm3 is None:
            Al_density_g_per_cm3=xraylib.ElementDensity(xraylib.SymbolToAtomicNumber('Al'))
        Al_area_density_g_per_cm2 = thickness_nm * Al_density_g_per_cm3 *1e-7    
        oxide_dict = xraylib.GetCompoundDataNISTByName('Aluminum Oxide')
        oxide_atomic_numbers = np.array(oxide_dict['Elements'])
        oxide_material = [xraylib.AtomicNumberToSymbol(iAtomicNumber) for iAtomicNumber in oxide_atomic_numbers]
        oxide_thickness = num_oxidized_surfaces * 3.0e-7 # cm
        if oxide_density_g_per_cm3 is None:
            oxide_density_g_per_cm3 = oxide_dict['density']
        oxide_mass_fractions = np.array(oxide_dict['massFractions'])
        oxide_area_density_g_per_cm2 = oxide_mass_fractions * oxide_density_g_per_cm3 * oxide_thickness
        material = np.hstack(['Al', oxide_material])
        area_density_g_per_cm2=np.hstack([Al_area_density_g_per_cm2, oxide_area_density_g_per_cm2])
        super().__init__(name='Al Film + Native Oxide', material=material, area_density_g_per_cm2=area_density_g_per_cm2)

class AlFilmWithPolymer(Film):
    def __init__(self, name, Al_thickness_nm, polymer_thickness_nm, Al_density_g_per_cm3=None, num_oxidized_surfaces=1, 
    oxide_density_g_per_cm3=None, polymer_fractions=None, polymer_density_g_per_cm3=None):
        assert num_oxidized_surfaces in [1,2], 'only 1 or 2 oxidzed surfaces allowed'
        if Al_density_g_per_cm3 is None:
            Al_density_g_per_cm3=xraylib.ElementDensity(xraylib.SymbolToAtomicNumber('Al'))
        Al_area_density_g_per_cm2 = Al_thickness_nm * Al_density_g_per_cm3 *1e-7    
        oxide_dict = xraylib.GetCompoundDataNISTByName('Aluminum Oxide')
        oxide_atomic_numbers = np.array(oxide_dict['Elements'])
        oxide_material = [xraylib.AtomicNumberToSymbol(iAtomicNumber) for iAtomicNumber in oxide_atomic_numbers]
        oxide_thickness = num_oxidized_surfaces * 3.0e-7 # cm
        if oxide_density_g_per_cm3 is None:
            oxide_density_g_per_cm3 = oxide_dict['density']
        oxide_mass_fractions = np.array(oxide_dict['massFractions'])
        oxide_area_density_g_per_cm2 = oxide_mass_fractions * oxide_density_g_per_cm3 * oxide_thickness
        polymer_material = ['H', 'C', 'N', 'O']
        polymer_density_g_per_cm3 = 1.4
        polymer_mass_fractions = np.array([0.0264, 0.6911, 0.0733, 0.2092])
        polymer_area_density_g_per_cm2 = polymer_mass_fractions * polymer_density_g_per_cm3 * polymer_thickness_nm * 1e-7
        material = np.hstack(['Al', oxide_material, polymer_material])
        area_density_g_per_cm2=np.hstack([Al_area_density_g_per_cm2, oxide_area_density_g_per_cm2, polymer_area_density_g_per_cm2])
        super().__init__(name='Al Film + Polymer', material=material, area_density_g_per_cm2=area_density_g_per_cm2)

class LEX_HT(FilterStack):
    def __init__(self, name):
        super().__init__(name)
        # Set up Al + polyimide film
        film_material = ['C', 'H', 'N', 'O', 'Al']
        film_area_density_g_per_cm2 = [6.7e-5, 2.6e-6, 7.2e-6, 1.7e-5, 1.7e-5]
        self.add_Film(name='LEX_HT Film', material=film_material, area_density_g_per_cm2=film_area_density_g_per_cm2)
        # Set up mesh
        mesh_material = ['Fe','Cr', 'Ni', 'Mn', 'Si']
        mesh_thickness = 100.0e-4 # cm
        mesh_density = 8.0 # g/cm^3
        mesh_material_fractions = np.array([0.705, 0.19, 0.09, 0.01, 0.005]) # fraction by weight
        mesh_area_density_g_per_cm2 = mesh_material_fractions * mesh_density * mesh_thickness # g/cm^2
        mesh_fill_fraction = 0.19
        self.add_Mesh(name='LEX_HT Mesh', material=mesh_material, area_density_g_per_cm2=mesh_area_density_g_per_cm2, fill_fraction=mesh_fill_fraction)

# EBIT Instrument
EBIT_filter_stack = FilterStack(name='EBIT Filter Stack 2018')
EBIT_filter_stack.add_Film(name='Electroplated Au Absorber', material='Au', thickness_nm=965.5, absorber=True)
EBIT_filter_stack.add_AlFilmWithOxide(name='50mK Filter', thickness_nm=112.5)
EBIT_filter_stack.add_AlFilmWithOxide(name='3K Filter', thickness_nm=108.5)
filter_50K = FilterStack(name='50K Filter')
filter_50K.add_AlFilmWithOxide(name='50K Filter',thickness_nm=102.6)
filter_50K.add_Mesh(name='Ni Mesh', material='Ni', thickness_nm=15.0e3, fill_fraction=0.17)
EBIT_filter_stack.add(filter_50K)
EBIT_filter_stack.add_LEX_HT('Luxel Window TES')
EBIT_filter_stack.add_LEX_HT('Luxel Window EBIT')
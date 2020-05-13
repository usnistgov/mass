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

class LEX_HT(FilterStack):
    def __init__(self, name):
        super().__init__(name)
        # Set up Al + polyimide film
        film_material = ['C', 'H', 'N', 'O', 'Al']
        film_area_density_g_per_cm2 = [6.7e-5, 2.6e-6, 7.2e-6, 1.7e-5, 1.7e-5]
        film_name = 'LEX_HT Film'
        self.add(Film(name=film_name, material=film_material, area_density_g_per_cm2=film_area_density_g_per_cm2))
        # Set up mesh
        mesh_material = ['Fe','Cr', 'Ni', 'Mn', 'Si']
        mesh_thickness = 100.0e-4 # cm
        mesh_density = 8.0 # g/cm^3
        mesh_material_fractions = np.array([0.705, 0.19, 0.09, 0.01, 0.005]) # fraction by weight
        mesh_area_density_g_per_cm2 = mesh_material_fractions * mesh_density * mesh_thickness # g/cm^2
        mesh_fill_fraction = 0.19
        mesh_name = 'LEX_HT Mesh'
        self.add(Mesh(name=mesh_name, material=mesh_material, area_density_g_per_cm2=mesh_area_density_g_per_cm2, fill_fraction=mesh_fill_fraction))

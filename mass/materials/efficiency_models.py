from uncertainties import ufloat
from uncertainties import unumpy as unp
from .uncertainties_helpers import ensure_uncertain, with_fractional_uncertainty
import numpy as np
import pylab as plt
import xraydb
from collections import OrderedDict


class FilterStack:
    def __init__(self, name):
        self.name = name
        self.components = OrderedDict()

    def add(self, c):
        assert isinstance(c, FilterStack)
        self.components[c.name] = c

    def add_Film(self, name, material, area_density_g_per_cm2=None, thickness_nm=None,
                 density_g_per_cm3=None, fill_fraction=ufloat(1, 0), absorber=False):
        c = Film(name=name, material=material, area_density_g_per_cm2=area_density_g_per_cm2,
                 thickness_nm=thickness_nm, density_g_per_cm3=density_g_per_cm3, fill_fraction=fill_fraction, absorber=absorber)
        self.components[c.name] = c

    def add_AlFilmWithOxide(self, name, Al_thickness_nm, Al_density_g_per_cm3=None,
                            num_oxidized_surfaces=2, oxide_density_g_per_cm3=None):
        c = AlFilmWithOxide(name=name, Al_thickness_nm=Al_thickness_nm, Al_density_g_per_cm3=Al_density_g_per_cm3,
                            num_oxidized_surfaces=num_oxidized_surfaces, oxide_density_g_per_cm3=oxide_density_g_per_cm3)
        self.components[c.name] = c

    def add_AlFilmWithPolymer(self, name, Al_thickness_nm, polymer_thickness_nm, polymer_fractions=None,
                              polymer_density_g_per_cm3=None,
                              num_oxidized_surfaces=1, oxide_density_g_per_cm3=None):
        c = AlFilmWithPolymer(name=name, Al_thickness_nm=Al_thickness_nm, polymer_thickness_nm=polymer_thickness_nm,
                              polymer_fractions=polymer_fractions, polymer_density_g_per_cm3=polymer_density_g_per_cm3,
                              num_oxidized_surfaces=num_oxidized_surfaces, oxide_density_g_per_cm3=oxide_density_g_per_cm3)
        self.components[c.name] = c

    def add_LEX_HT(self, name):
        c = LEX_HT(name=name)
        self.components[c.name] = c

    def get_efficiency(self, xray_energies_eV, uncertain=False):
        assert self.components != {
        }, f'{self.name} has no components of which to calculate efficiency'
        individual_efficiency = np.array([iComponent.get_efficiency(
            xray_energies_eV, uncertain=uncertain) for iComponent in list(self.components.values())])
        efficiency = np.prod(individual_efficiency, axis=0)
        if uncertain:
            return efficiency
        else:
            return unp.nominal_values(efficiency)

    def __call__(self, xray_energies_eV, uncertain=False):
        return self.get_efficiency(xray_energies_eV, uncertain=uncertain)

    def plot_efficiency(self, xray_energies_eV, ax=None):
        efficiency = unp.nominal_values(self.get_efficiency(xray_energies_eV))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(xray_energies_eV, efficiency*100.0, label="total", lw=2)
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title(self.name)
        ax.set_title(f'{self.name} Efficiency')

        for k, v in self.components.items():
            efficiency = v.get_efficiency(xray_energies_eV)
            ax.plot(xray_energies_eV, efficiency*100.0, "--", label=v.name)

        ax.legend()

    def __repr__(self):
        s = f"{type(self)}(\n"
        for k, v in self.components.items():
            s += f"{k}: {v}\n"
        s += ")"
        return s


class Film(FilterStack):
    def __init__(self, name, material, area_density_g_per_cm2=None, thickness_nm=None,
                 density_g_per_cm3=None, fill_fraction=1.0, absorber=False):
        """
        Must construct with an areal density. Options:
        1. Give areal density directly as `area_density_g_per_cm2`, or
        2. Give thickness (nm) as `thickness_nm`.

        In case 2, the areal density is density times thickness.
        In case 1, thickness will be computed as areal density / density just for reference (because)
        areal density is what actually controls the film's attenuation/transmission behavior.

        If density isn't given (as `density_g_per_cm3`), a standard value for the material will be
        assumed. This value affects the filter transmission only in case 2.
        """
        super().__init__(name)
        self.material = np.array(material, ndmin=1)
        self.atomic_number = np.array([xraydb.atomic_number(iMaterial)
                                       for iMaterial in self.material], ndmin=1)
        self.fill_fraction = ensure_uncertain(fill_fraction)
        self.absorber = absorber

        # Save density, either default values for that element, or the given density.
        if density_g_per_cm3 is None:
            self.density_g_per_cm3 = np.array([xraydb.atomic_density(
                int(iAtomicNumber)) for iAtomicNumber in self.atomic_number], ndmin=1)
        else:
            self.density_g_per_cm3 = np.array(density_g_per_cm3, ndmin=1)
            assert len(self.material) == len(self.density_g_per_cm3)

        # Handle input value of areal density or thickness, but not both.
        assert np.logical_xor(area_density_g_per_cm2 is None,
                              thickness_nm is None), 'must specify either areal density or thickness, not both'
        if thickness_nm is not None:
            thickness_cm = np.array(thickness_nm, ndmin=1)*1e-7
        elif area_density_g_per_cm2 is not None:
            area_density_g_per_cm2 = np.array(area_density_g_per_cm2, ndmin=1)
            thickness_cm = area_density_g_per_cm2 / self.density_g_per_cm3
            if np.ndim == 0:
                thickness_cm = np.array(thickness_cm, ndmin=1)
        else:
            raise ValueError('must specify either areal density or thickness, not both')
        self.thickness_cm = ensure_uncertain(thickness_cm)
        assert len(self.thickness_cm) >= 1

    def get_efficiency(self, xray_energies_eV, uncertain=False):
        optical_depth = np.vstack([
            xraydb.material_mu(m, xray_energies_eV, density=d) * t
            for (m, d, t) in zip(self.material, self.density_g_per_cm3, self.thickness_cm)])
        individual_transmittance = unp.exp(-optical_depth)
        transmittance = np.prod(individual_transmittance, axis=0)
        if self.absorber:
            efficiency = (1.0 - transmittance) * self.fill_fraction
        else:
            efficiency = (transmittance * self.fill_fraction) + (1.0 - self.fill_fraction)
        if uncertain:
            return efficiency
        else:
            return unp.nominal_values(efficiency)

    def __repr__(self):
        s = f"{type(self)}("
        for (material, density, thick) in zip(self.material, self.density_g_per_cm3, self.thickness_cm):
            area_density = density*thick
            s += f"{material} {area_density:.3g} g/cm^2, "
        s += f"fill_fraction={self.fill_fraction:.3f}, absorber={self.absorber})"
        return s


class AlFilmWithOxide(Film):
    ''' Create an Al film with 3nm Al2O3 oxide on both sides.

    Ideal for modeling free-standing Ir-blocking Al filters. '''

    def __init__(self, name, Al_thickness_nm, Al_density_g_per_cm3=None, num_oxidized_surfaces=2, oxide_density_g_per_cm3=None):
        ''' Initialize AlFilmWithOxide object

        Args:
            name: name given to filter object, e.g. '50K Filter'.
            Al_thickness_nm: thickness, in nm, of Al film
            Al_density_g_per_cm3: Al film density, in g/cm3, defaults to xraydb value
            num_oxidized_surfaces: Number of film surfaces that contain a native oxide, default 2
            oxide_density_g_per_cm3: Al2O3 oxide density, in g/cm3, defaults to bulk xraydb value

        '''
        assert num_oxidized_surfaces in [1, 2], 'only 1 or 2 oxidzed surfaces allowed'
        if Al_density_g_per_cm3 is None:
            Al_density_g_per_cm3 = xraydb.atomic_density('Al')
        arbE = 5000.  # an arbitrary energy (5 keV) is used to get answers from material_mu_components()
        oxide_dict = xraydb.material_mu_components('sapphire', arbE)
        oxide_material = oxide_dict['elements']
        oxide_mass_fractions = [oxide_dict[x][0]*oxide_dict[x]
                                [1]/oxide_dict['mass'] for x in oxide_material]

        # Assume oxidized surfaces are each 3 nm thick.
        num_oxide_elements = len(oxide_material)
        oxide_thickness_nm = np.repeat(num_oxidized_surfaces * 3.0, num_oxide_elements)
        if oxide_density_g_per_cm3 is None:
            oxide_density_g_per_cm3 = np.repeat(oxide_dict['density'], num_oxide_elements)

        material = np.hstack(['Al', oxide_material])
        density_g_per_cm3 = np.hstack(
            [Al_density_g_per_cm3, oxide_density_g_per_cm3*oxide_mass_fractions])
        thickness_nm = np.hstack([Al_thickness_nm, oxide_thickness_nm])
        super().__init__(name=name, material=material,
                         density_g_per_cm3=density_g_per_cm3, thickness_nm=thickness_nm)


class AlFilmWithPolymer(Film):
    ''' Create an Al film with polymer layer on 1 side and native oxide on other.

    Ideal for modeling polymer-backed IR-blocking aluminum filters. '''

    def __init__(self, name, Al_thickness_nm, polymer_thickness_nm, Al_density_g_per_cm3=None, num_oxidized_surfaces=1,
                 oxide_density_g_per_cm3=None, polymer_fractions=None, polymer_density_g_per_cm3=None):
        ''' Initialize AlFilmWithPolymer object

        Args:
            name: name given to filter object, e.g. '50K Filter'.
            Al_thickness_nm: thickness, in nm, of Al film
            polymer_thickness_nm: thickness, in nm, of filter backside polymer
            Al_density_g_per_cm3: Al film density, in g/cm3, defaults to xraydb value
            num_oxidized_surfaces: Number of film surfaces that contain a native oxide, default 2
            oxide_density_g_per_cm3: Al2O3 oxide density, in g/cm3, defaults to bulk xraydb value
            polymer_fractions: elemental mass fractions of polymer used, defaults to Kapton
            polymer_density_g_per_cm3: Polymer density, in g/cm3, defaults to Kapton
        '''
        assert num_oxidized_surfaces in [1, 2], 'only 1 or 2 oxidzed surfaces allowed'
        if Al_density_g_per_cm3 is None:
            Al_density_g_per_cm3 = xraydb.atomic_density('Al')

        arbE = 5000.  # an arbitrary energy (5 keV) is used to get answers from material_mu_components()
        oxide_dict = xraydb.material_mu_components('sapphire', arbE)
        oxide_thickness_nm = num_oxidized_surfaces * 3.0  # assume 3 nm per oxidized surface
        oxide_material = oxide_dict['elements']
        oxide_mass_fractions = [oxide_dict[x][0]*oxide_dict[x]
                                [1]/oxide_dict['mass'] for x in oxide_material]
        if oxide_density_g_per_cm3 is None:
            oxide_density_g_per_cm3 = oxide_dict['density']*np.ones(len(oxide_material))

        polymer_dict = xraydb.material_mu_components('kapton', arbE)
        polymer_material = polymer_dict['elements']
        polymer_thickness_nm = polymer_thickness_nm*np.ones(len(polymer_material))
        polymer_mass_fractions = [polymer_dict[x][0]*polymer_dict[x][1]
                                  / polymer_dict['mass'] for x in polymer_material]
        if polymer_density_g_per_cm3 is None:
            polymer_density_g_per_cm3 = polymer_dict['density']*np.ones(len(polymer_material))

        material = np.hstack(['Al', oxide_material, polymer_material])
        density_g_per_cm3 = np.hstack([
            Al_density_g_per_cm3,
            oxide_density_g_per_cm3*oxide_mass_fractions,
            polymer_density_g_per_cm3*polymer_mass_fractions])
        thickness_nm = np.hstack([Al_thickness_nm, oxide_thickness_nm, polymer_thickness_nm])

        super().__init__(name=name, material=material,
                         density_g_per_cm3=density_g_per_cm3, thickness_nm=thickness_nm)


class LEX_HT(FilterStack):
    ''' Create an Al film with polymer and stainless steel backing.

    Ideal modeling LEX-HT vacuum window.'''

    def __init__(self, name):
        ''' Initialize LEX_HT object

        Args:
            name: name given to filter object, e.g. '50K Filter'.
        '''
        super().__init__(name)
        # Set up Al + polyimide film
        film_material = ['C', 'H', 'N', 'O', 'Al']
        film_area_density_g_per_cm2_given = np.array([6.7e-5, 2.6e-6, 7.2e-6, 1.7e-5, 1.7e-5])
        film_area_density_g_per_cm2 = with_fractional_uncertainty(
            film_area_density_g_per_cm2_given, 0.03)
        self.add_Film(name='LEX_HT Film', material=film_material,
                      area_density_g_per_cm2=film_area_density_g_per_cm2)
        # Set up mesh
        mesh_material = ['Fe', 'Cr', 'Ni', 'Mn', 'Si']
        mesh_thickness = 100.0e-4  # cm
        mesh_density = 8.0  # g/cm^3
        mesh_material_fractions = np.array([0.705, 0.19, 0.09, 0.01, 0.005])  # fraction by weight
        mesh_area_density_g_per_cm2_scalar = mesh_material_fractions * mesh_density * mesh_thickness  # g/cm^2
        mesh_area_density_g_per_cm2 = with_fractional_uncertainty(
            mesh_area_density_g_per_cm2_scalar, 0.02)
        mesh_fill_fraction = ufloat(0.19, 0.01)
        self.add_Film(name='LEX_HT Mesh', material=mesh_material,
                      area_density_g_per_cm2=mesh_area_density_g_per_cm2, fill_fraction=mesh_fill_fraction)


def get_filter_stacks_dict():
    # Create models for TES instruments
    fs_dict = {}

    # EBIT Instrument
    EBIT_filter_stack = FilterStack(name='EBIT 2018')
    EBIT_filter_stack.add_Film(name='Electroplated Au Absorber',
                               material='Au', thickness_nm=with_fractional_uncertainty(965.5, 0.03), absorber=True)
    EBIT_filter_stack.add_AlFilmWithOxide(
        name='50mK Filter', Al_thickness_nm=with_fractional_uncertainty(112.5, 0.02))
    EBIT_filter_stack.add_AlFilmWithOxide(
        name='3K Filter', Al_thickness_nm=with_fractional_uncertainty(108.5, 0.02))
    filter_50K = FilterStack(name='50K Filter')
    filter_50K.add_AlFilmWithOxide(
        name='Al Film', Al_thickness_nm=with_fractional_uncertainty(102.6, 0.02))
    filter_50K.add_Film(name='Ni Mesh', material='Ni', thickness_nm=ufloat(
        15.0e3, 2e3), fill_fraction=ufloat(0.17, 0.01))
    EBIT_filter_stack.add(filter_50K)
    EBIT_filter_stack.add_LEX_HT('Luxel Window TES')
    EBIT_filter_stack.add_LEX_HT('Luxel Window EBIT')
    fs_dict[EBIT_filter_stack.name] = EBIT_filter_stack

    # RAVEN Instrument
    RAVEN1_fs = FilterStack(name='RAVEN1 2019')
    RAVEN1_fs.add_Film(name='Evaporated Bi Absorber', material='Bi',
                       thickness_nm=4.4e3, absorber=True)
    RAVEN1_fs.add_AlFilmWithPolymer(
        name='50mK Filter', Al_thickness_nm=108.4, polymer_thickness_nm=206.4)
    RAVEN1_fs.add_AlFilmWithPolymer(
        name='3K Filter', Al_thickness_nm=108.4, polymer_thickness_nm=206.4)
    RAVEN1_fs.add_AlFilmWithOxide(name='50K Filter', Al_thickness_nm=1.0e3)
    RAVEN1_fs.add_Film(name='Be TES Vacuum Window', material='Be', thickness_nm=200.0e3)
    RAVEN1_fs.add_AlFilmWithOxide(name='e- Filter', Al_thickness_nm=5.0e3)
    RAVEN1_fs.add_Film(name='Be SEM Vacuum Window', material='Be', thickness_nm=200.0e3)
    fs_dict[RAVEN1_fs.name] = RAVEN1_fs

    # Horton spring 2018, for metrology campaign.
    Horton_filter_stack = FilterStack(name='Horton 2018')
    Horton_filter_stack.add_Film(name='Electroplated Au Absorber',
                                 material='Au', thickness_nm=965.5, absorber=True)
    Horton_filter_stack.add_AlFilmWithOxide(name='50mK Filter', Al_thickness_nm=5000)
    Horton_filter_stack.add_AlFilmWithOxide(name='3K Filter', Al_thickness_nm=5000)
    Horton_filter_stack.add_AlFilmWithOxide(name='50K Filter', Al_thickness_nm=12700)
    Horton_filter_stack.add_LEX_HT('Luxel Window TES')
    fs_dict[Horton_filter_stack.name] = Horton_filter_stack

    return fs_dict


filterstack_models = get_filter_stacks_dict()

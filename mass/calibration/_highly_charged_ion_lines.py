from . import fluorescence_lines
from . import line_models
from . import LORENTZIAN_PEAK_HEIGHT
import numpy as np

'''
O LINES
'''
# H-like
fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" H-Like 2p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(653.679946*2+653.493657*1)/3,
    energies=np.array([653.493657, 653.679946]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None
)

fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" H-Like 3p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(774.634043*2+774.578843*1)/3,
    energies=np.array([774.634043, 774.578843]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([2, 1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None
)

fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" H-Like 4p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(816.974368*2+816.951082*1)/3,
    energies=np.array([816.951082, 816.974368]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([2, 1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None
)

# He-like
fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2s+1s2p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=573.94777,
    energies=np.array([560.983, 568.551, 573.94777]), lorentzian_fwhm=np.array([0.1, 0.1, 0.1]),
    reference_amplitude=np.array([500, 300, 1000]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2s 3S1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=560.98386,
    energies=np.array([560.98386]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=573.94777,
    energies=np.array([573.94777]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" He-Like 1s3p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=665.61536,
    energies=np.array([665.61536]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="O",
    material="Highly Charged Ion",
    linetype=" He-Like 1s4p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=697.79546,
    energies=np.array([697.79546]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

'''
Ne LINES
'''
# H-like
fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" H-Like 2p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(1021.952896*2+1021.497550*1)/3,
    energies=np.array([1021.497550, 1021.952896]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" H-Like 3p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(1210.961453*2+1210.826524*1)/3,
    energies=np.array([1210.826524, 1210.961453]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" H-Like 4p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(1277.130058*2+1277.073140*1)/3,
    energies=np.array([1277.073140, 1277.130058]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" H-Like 5p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(1307.757092*2+1307.727954*1)/3,
    energies=np.array([1307.727954, 1307.757092]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" H-Like 6p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(1324.393513*2+1324.376653*1)/3,
    energies=np.array([1324.376653, 1324.393513]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" H-Like 7p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(1334.424384*2+1334.413767*1)/3,
    energies=np.array([1334.413767, 1334.424384]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

# He-like
fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2s+1s2p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=922.0159,
    energies=np.array([905.0772, 914.8174, 922.0159]), lorentzian_fwhm=np.array([0.1, 0.1, 0.1]),
    reference_amplitude=np.array([500, 150, 1000]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2s 3S1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=905.0772,
    energies=np.array([905.0772]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=922.0159,
    energies=np.array([922.0159]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="Ne",
    material="Highly Charged Ion",
    linetype=" He-Like 1s3p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1073.7689,
    energies=np.array([1073.7689]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

'''
Ar LINES
'''
# H-like
fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" H-Like 2p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(3322.9921*2+3318.1762*1)/3,
    energies=np.array([3318.1762, 3322.9921]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" H-Like 3p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(3935.72070*2+3934.29336*1)/3,
    energies=np.array([3934.29336, 3935.72070]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" H-Like 4p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=(4150.33999*2+4149.73807*1)/3,
    energies=np.array([4149.73807, 4150.33999]), lorentzian_fwhm=np.array([0.1, 0.1]),
    reference_amplitude=np.array([1, 2]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

# He-like
fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2s+1s2p",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3139.5824,
    energies=np.array([3104.1486, 3123.5346, 3139.5824]), lorentzian_fwhm=np.array([0.1, 0.1, 0.1]),
    reference_amplitude=np.array([100, 55, 200]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2s 3S1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3104.1486,
    energies=np.array([3104.1486]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" He-Like 1s2p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3139.5824,
    energies=np.array([3139.5824]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" He-Like 1s3p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3683.848,
    energies=np.array([3683.848]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

fluorescence_lines.addline(
    element="Ar",
    material="Highly Charged Ion",
    linetype=" He-Like 1s4p 1P1",
    reference_short='NIST ASD',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3874.886,
    energies=np.array([3874.886]), lorentzian_fwhm=np.array([0.1]),
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Theory"
)

'''
W Lines
'''
# Ni-like
fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-1",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1488.2,
    energies=np.array([1488.2]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-2",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1562.9,
    energies=np.array([1562.9]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-3",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1629.8,
    energies=np.array([1629.8]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-4",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1764.6,
    energies=np.array([1764.6]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-5",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1829.6,
    energies=np.array([1829.6]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-6",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2015.4,
    energies=np.array([2015.4]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-7",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2112.2,
    energies=np.array([2112.2]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-8",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2179.7,
    energies=np.array([2179.7]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-9",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2320.3,
    energies=np.array([2320.3]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.6,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-10",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2360.7,
    energies=np.array([2360.7]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.7,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-11",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2384.2,
    energies=np.array([2384.2]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-12",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2553.0,
    energies=np.array([2553.0]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-13",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2651.3,
    energies=np.array([2651.3]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-14",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2673.7,
    energies=np.array([2673.7]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.6,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-15",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2760.7,
    energies=np.array([2760.7]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.5,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-16",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2816.1,
    energies=np.array([2816.1]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-17",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=2878.2,
    energies=np.array([2878.2]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-18",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3182.7,
    energies=np.array([3182.7]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-19",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3196.8,
    energies=np.array([3196.8]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-20",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3259.9,
    energies=np.array([3259.9]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.3,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-21",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3426.0,
    energies=np.array([3426.0]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-22",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3480.9,
    energies=np.array([3480.9]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.7,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-23",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3490.2,
    energies=np.array([3490.2]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.4,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-24",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3574.1,
    energies=np.array([3574.1]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.5,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-25",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3600.0,
    energies=np.array([3600.0]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.6,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-26",
    reference_short='Clementson 2010',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=3639.5,
    energies=np.array([3639.5]), lorentzian_fwhm=np.array([0.1]),
    position_uncertainty=0.6,
    reference_amplitude=np.array([1]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

fluorescence_lines.addline(
    element="W",
    material="Highly Charged Ion",
    linetype=" Ni-Like 3s^2,3p^6,3s^3_3/2,3d^6_5/2,4p_1/2",
    reference_short='Nilsen 1995',
    fitter_type=line_models.GenericLineModel,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=1728.41,
    energies=np.array([1725.04, 1728.41]), lorentzian_fwhm=np.array([0.1, 0.1]),
    position_uncertainty=0.11,
    reference_amplitude=np.array([1, 20]),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None,
    reference_measurement_type="Experiment"
)

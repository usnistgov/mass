"""
hci_models.py

Some useful methods for initializing GenericLineModel and CompositeMLEModel objects applied to HCI lines.

June 2020
Paul Szypryt
"""

import numpy as np
import mass.calibration.hci_lines
import xraydb



def initialize_hci_line_model(line_name, has_linear_background=False, has_tails=False):
    '''Initializes a single lorentzian hci lmfit model. Reformats line_name to create a lmfit valid prefix.

    Args:
        line_name: name of line within mass.spectrum_classes
        has_linear_background: (default False) include linear background in the model
        has_tails: (default False) include low energy tail in the model
    '''

    line = mass.spectrum_classes[line_name]()
    prefix = f'{line_name}_'.replace(' ', '_').replace(
        'J=', '').replace('/', '_').replace('*', '').replace('.', '')
    line_model = line.model(has_linear_background=has_linear_background, has_tails=has_tails, prefix=prefix)
    line_model.shortname = line_name
    return line_model


def initialize_hci_composite_model(composite_name, individual_models, has_linear_background=False, peak_component_name=None):
    '''Initializes composite lmfit model from the sum of input models

    Args:
        composite_name: str name given to composite line model
        line_models: array of lmfit models to sum into a composite model
        has_linear_background: (default False) include a single linear background on top of group of lorentzians
        peak_component_name: designate a component to be a peak for energy, all expressions are referenced to this component
    '''

    composite_model = np.sum(individual_models)
    composite_model.name = composite_name
    if has_linear_background:
        composite_model = add_bg_model(composite_model)
    # Workaround for energy calibration using composite models, pick 1st GenericLineModel component
    line_model_components = [i_comp for i_comp in composite_model.components if isinstance(
        i_comp, mass.calibration.line_models.GenericLineModel)]
    if peak_component_name is None:
        peak_component_name = line_model_components[0]._name
    peak_component_index = [i_comp._name for i_comp in line_model_components].index(peak_component_name)
    peak_component = line_model_components[peak_component_index]
    composite_model.peak_prefix = peak_component.prefix
    composite_model.peak_energy = peak_component.spect.peak_energy
    # Set up some constraints relative to peak_component
    num_line_components = len(line_model_components)
    line_component_prefixes = [iComp.prefix for iComp in line_model_components]
    line_component_energies = [iComp.spect.peak_energy for iComp in line_model_components]
    for i in np.arange(num_line_components):
        if i != peak_component_index:
            # Single fwhm across model
            composite_model.set_param_hint(f'{line_component_prefixes[i]}fwhm',
                                           expr=f'{composite_model.peak_prefix}fwhm')
            # Single dph_de across model
            composite_model.set_param_hint(f'{line_component_prefixes[i]}dph_de',
                                           expr=f'{composite_model.peak_prefix}dph_de')
            # Fixed energy separation based on database values
            separation = line_component_energies[i] - composite_model.peak_energy
            composite_model.set_param_hint(f'{line_component_prefixes[i]}peak_ph',
                                           expr='({0} * {1}dph_de) + {1}peak_ph'.format(separation, composite_model.peak_prefix))
    composite_model.shortname = composite_name
    return composite_model


def initialize_HLike_2P_model(element, conf, has_linear_background=False, has_tails=False, vary_amp_ratio=False):
    '''Initializes H-like 2P models consisting of J=1/2 and J=3/2 lines

    Args:
        element: atomic symbol as str, e.g. 'Ne' or 'Ar'
        conf: nuclear configuration as str, e.g. '2p' or '3p'
        has_linear_background: (default False) include a single linear background on top of the 2 Lorentzians
        has_tails: (default False) include low energy tail in the model
        vary_amp_ratio: (default False) allow the ratio of the J=3/2 to J=1/2 states to vary away from 2
    '''

    # Set up line names and lmfit prefixes
    charge = int(xraydb.atomic_number(element))
    line_name_1_2 = f'{element}{charge} {conf} 2P* J=1/2'
    line_name_3_2 = f'{element}{charge} {conf} 2P* J=3/2'
    prefix_1_2 = f'{line_name_1_2}_'.replace(' ', '_').replace(
        'J=', '').replace('/', '_').replace('*', '').replace('.', '')
    prefix_3_2 = f'{line_name_3_2}_'.replace(' ', '_').replace(
        'J=', '').replace('/', '_').replace('*', '').replace('.', '')
    # Initialize individual lines and models
    line_1_2 = mass.spectrum_classes[line_name_1_2]()
    line_3_2 = mass.spectrum_classes[line_name_3_2]()
    model_1_2 = line_1_2.model(has_linear_background=False, has_tails=has_tails, prefix=prefix_1_2)
    model_3_2 = line_3_2.model(has_linear_background=False, has_tails=has_tails, prefix=prefix_3_2)
    # Initialize composite model and set addition H-like constraints
    composite_name = f'{element}{charge} {conf}'
    composite_model = initialize_hci_composite_model(composite_name=composite_name, individual_models=[model_1_2, model_3_2],
                                                     has_linear_background=has_linear_background, peak_component_name=line_name_3_2)
    amp_ratio_param_name = f'{element}{charge}_{conf}_amp_ratio'
    composite_model.set_param_hint(name=amp_ratio_param_name, value=0.5, min=0.0, vary=vary_amp_ratio)
    composite_model.set_param_hint(f'{prefix_1_2}integral', expr=f'{prefix_3_2}integral * {amp_ratio_param_name}')
    return composite_model


def initialize_HeLike_complex_model(element, has_linear_background=False, has_tails=False, additional_line_names=[]):
    '''Initializes 1s2s,2p He-like complexes for a given element.
    By default, uses only the 1s.2s 3S J=1, 1s.2p 3P* J=1, and 1s.2p 1P* J=1 lines.

    Args:
        element: atomic symbol as str, e.g. 'Ne' or 'Ar'
        has_linear_background: (default False) include a single linear background on top of the Lorentzian models
        has_tails: (default False) include low energy tail in the model
        additional_line_names: (default []) additional line names to include in model, e.g. low level Li/Be-like features
    '''

    # Set up line names
    charge = int(xraydb.atomic_number(element) - 1)
    line_name_1s2s_3S = f'{element}{charge} 1s.2s 3S J=1'
    line_name_1s2p_3P = f'{element}{charge} 1s.2p 3P* J=1'
    line_name_1s2p_1P = f'{element}{charge} 1s.2p 1P* J=1'
    line_names = np.hstack([[line_name_1s2s_3S, line_name_1s2p_3P, line_name_1s2p_1P], additional_line_names])
    # Set up lines and models based on line_names
    # individual_lines = [mass.spectrum_classes[i_line_name]() for i_line_name in line_names]
    individual_models = [initialize_hci_line_model(
        i_line_name, has_linear_background=False, has_tails=has_tails) for i_line_name in line_names]
    # Set up composite model
    composite_name = f'{element}{charge} 1s2s_2p Complex'
    composite_model = initialize_hci_composite_model(
        composite_name=composite_name, individual_models=individual_models,
        has_linear_background=has_linear_background, peak_component_name=line_name_1s2p_1P)
    return composite_model


def add_bg_model(generic_model, vary_slope=False):
    '''Adds a LinearBackgroundModel to a generic lmfit model

    Args:
        generic_model: Generic lmfit model object to which to add a linear background model
        vary_slope: (default False) allows a varying linear slope rather than just constant value
    '''

    composite_name = generic_model._name
    bg_prefix = f'{composite_name}_'.replace(' ', '_').replace(
        'J=', '').replace('/', '_').replace('*', '').replace('.', '')
    background_model = mass.calibration.line_models.LinearBackgroundModel(
        name=f'{composite_name} Background', prefix=bg_prefix)
    background_model.set_param_hint('bg_slope', vary=vary_slope)
    composite_model = generic_model + background_model
    composite_model.name = composite_name
    return composite_model


def models(has_linear_background=False, has_tails=False, vary_Hlike_amp_ratio=False, additional_Helike_complex_lines=[]):
    '''
    Generates some commonly used HCI line models that can be used for energy calibration, etc.

    Args:
        has_linear_background: (default False) include a single linear background on top of the 2 Lorentzians
        has_tails: (default False) include low energy tail in the model
        vary_Hlike_amp_ratio: (default False) allow the ratio of the J=3/2 to J=1/2 H-like states to vary
            additional_Helike_complex_lines: (default []) additional line names to include inHe-like complex
            model, e.g. low level Li/Be-like features
    '''

    models_dict = {}
    # Make some common H-like 2P* models
    conf_Hlike_2P_dict = {}
    conf_Hlike_2P_dict['N'] = ['3p', '4p', '5p']
    conf_Hlike_2P_dict['O'] = ['3p', '4p', '5p']
    conf_Hlike_2P_dict['Ne'] = ['2p', '3p', '4p', '5p']
    conf_Hlike_2P_dict['Ar'] = ['2p', '3p', '4p', '5p']
    for i_element in list(conf_Hlike_2P_dict.keys()):
        for i_conf in conf_Hlike_2P_dict[i_element]:
            Hlike_model = initialize_HLike_2P_model(i_element, i_conf, has_linear_background=has_linear_background,
                                                    has_tails=has_tails, vary_amp_ratio=vary_Hlike_amp_ratio)
            models_dict[Hlike_model._name] = Hlike_model

    # Make some common He-like 1s2s,2p complex and higher order 1p* models
    # He-like lines
    Helike_complex_elements = ['N', 'O', 'Ne', 'Ar']
    for i_element in Helike_complex_elements:
        Helike_model = initialize_HeLike_complex_model(
            i_element, has_linear_background=has_linear_background,
            has_tails=has_tails, additional_line_names=additional_Helike_complex_lines)
        models_dict[Helike_model._name] = Helike_model
    # 1s.np 1P* lines for n>=3
    conf_Helike_1P_dict = {}
    conf_Helike_1P_dict['N'] = ['1s.4p', '1s.5p']
    conf_Helike_1P_dict['O'] = ['1s.4p', '1s.5p']
    conf_Helike_1P_dict['Ne'] = ['1s.3p', '1s.4p', '1s.5p']
    conf_Helike_1P_dict['Ar'] = ['1s.3p', '1s.4p', '1s.5p']
    for i_element in list(conf_Helike_1P_dict.keys()):
        i_charge = int(xraydb.atomic_number(i_element) - 1)
        for i_conf in conf_Helike_1P_dict[i_element]:
            Helike_line_name = f'{i_element}{i_charge} {i_conf} 1P* J=1'
            Helike_model = initialize_hci_line_model(
                Helike_line_name, has_linear_background=has_linear_background, has_tails=has_tails)
            models_dict[Helike_model._name] = Helike_model

    # Some more complicated cases
    # 500 eV region of H-/He-like N
    N6_1s3p_model = initialize_hci_line_model(
        'N6 1s.3p 1P* J=1', has_linear_background=False, has_tails=has_tails)
    N7_2p_model = initialize_HLike_2P_model(
        'N', '2p', has_linear_background=False, has_tails=has_tails, vary_amp_ratio=vary_Hlike_amp_ratio)
    N_500eV_model = initialize_hci_composite_model('N 500eV Region', [N6_1s3p_model, N7_2p_model],
                                                   has_linear_background=has_linear_background, peak_component_name='N7 2p 2P* J=3/2')
    models_dict[N_500eV_model._name] = N_500eV_model
    # 660 eV region of H-/He-like O
    O8_2p_model = initialize_HLike_2P_model(
        'O', '2p', has_linear_background=False, has_tails=has_tails, vary_amp_ratio=vary_Hlike_amp_ratio)
    O7_1s3p_model = initialize_hci_line_model(
        'O7 1s.3p 1P* J=1', has_linear_background=False, has_tails=has_tails)
    O_660eV_model = initialize_hci_composite_model('O 660eV Region', [O8_2p_model, O7_1s3p_model],
                                                   has_linear_background=has_linear_background, peak_component_name='O8 2p 2P* J=3/2')
    models_dict[O_660eV_model._name] = O_660eV_model

    return models_dict

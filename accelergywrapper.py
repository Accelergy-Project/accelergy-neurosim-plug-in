""" Neurosim Accelergy Wrapper """
from accelergy.plug_in_interface.estimator_wrapper import (
    AccelergyPlugIn,
    Estimation,
    AccuracyEstimation,
    AccelergyQuery
)
import math
import sys
import os
from typing import Dict
# Need to add this directory to path for proper imports
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(SCRIPT_DIR)
# fmt: off
import neurointerface
# fmt: on

# ==================================================================================================
# Constants
# ==================================================================================================
DEBUG = False
ACCURACY = 70
PERMITTED_TECH_NODES = [7, 10, 14, 22, 32, 45, 65, 90, 130]
SAMPLE_CELLS = [
    f.split('.')[0] for f in os.listdir(os.path.join(SCRIPT_DIR, 'cells')) if f.endswith('.cell')]

CACHE = {}

# Format:
#   Key: (Docstring, Default value)
#   If REQUIRED in docstring, then a value is required. Otherwise, default is populated.
#   If a parameter is not used in an estimation (e.g. creating a shift+add), then the defaults will
#   be used for the non-selected device (PIM_PARAMS defaults for rows & columns are used).
PIM_PARAMS = {
    'rows':                    ('REQUIRED: Number of rows in a crossbar.', 32),
    'cols':                     ('REQUIRED: Number of columns in a crossbar.', 32),
    'cols_active_at_once':      ('REQUIRED: Number of columns active at once.', 8),
    'cell_config':              (f'REQUIRED: Path to a NVSim cell config file to use, or one of '
                                 f'the following samples: ' +
                                 f', '.join(
                                     f'"{s}"' for s in SAMPLE_CELLS), 'placeholder',
                                 str),
    'average_input_value':      (f' REQUIRED: Average input value to a row. Must be between '
                                 f'0 and 1.', 1, float),
    'average_cell_value':       (f' REQUIRED: Average cell value. Must be between 0 and 1.', 1,
                                 float),
    'latency':                  (f'REQUIRED: Latency between two subsequent reads of a cell in ns',
                                 1e-7, float),
    'sequential':               (f'OPTIONAL: Sequential mode. Default is False. If True, the '
                                 f'crossbar will be set up to activate one row at a time. '
                                 f'Can be used as a memory this way.', False),
    'adc_resolution':           (f'OPTIONAL: ADC resolution. Set this if to use Neurosim\'s '
                                 f'build-in ADC. Default is False.', 0),
    'read_pulse_width':         (f'OPTIONAL: Read pulse width. Default is 10ns.', 1e-8, float),

    'voltage_dac_bits':        (f'OPTIONAL: Resolution of a voltage DAC for inputs.', 1, int),
    'temporal_dac_bits':       (f'OPTIONAL: Resolution of a temporal DAC for inputs.', 1, int),
    'temporal_spiking':        (f'OPTIONAL: Whether to use a spiking (#pulses) or a PWM (pulse  '
                                f'length) approach for temporal DAC. Default is True ', True, bool),
    'adc_action_share':        (f'OPTIONAL: Manual scaling of ADC energy/op. Use this to simulate'
                                f' an ADC more or less efficient than Neurosim\'s build-in ADC. ',
                                1, float),
    'adc_area_share':          (f'OPTIONAL: Manual scaling of ADC area. Use this to simulate'
                                f' an ADC larger or smaller than Neurosim\'s build-in ADC. ',
                                1, float),
}

ADDER_PARAMS = {
    'n_bits':                   (f'REQUIRED: # Bits of the adder.', 8),
}
SHIFT_ADD_PARAMS = {
    'n_bits':                   (f'REQUIRED: # Bits of the adder.', 8),
    'shift_register_n_bits':    (f'REQUIRED: # Bits of the shift register.', 16),
}
MAX_POOL_PARAMS = {
    'n_bits':                   (f'REQUIRED: # Bits.', 8),
    'pool_window':              (f'REQUIRED: Window size of max pooling.', 2),
}
ADDER_TREE_PARAMS = {
    'n_bits':                   (f'REQUIRED: # Bits of the leaf adder.', 8),
    'n_adder_tree_inputs':      (f'REQUIRED: Number of inputs to the adder tree.', 2),
}
MUX_PARAMS = {
    'n_mux_inputs':             (f'REQUIRED: Number of inputs to the mux.', 2),
    'n_bits':                   (f'REQUIRED: # Bits of the mux.', 8),
}
FLIP_FLOP_PARAMS = {
    'n_bits':                   (f'REQUIRED: # Bits of flip-flop.', 8),
}
PARAM_DICTS = [
    PIM_PARAMS, ADDER_PARAMS, SHIFT_ADD_PARAMS,
    MAX_POOL_PARAMS, ADDER_TREE_PARAMS, MUX_PARAMS, FLIP_FLOP_PARAMS
]
for p in PARAM_DICTS:
    p['technology'] = (f'REQUIRED: Technology node. Must be between {max(PERMITTED_TECH_NODES)}  '
                       f'and {min(PERMITTED_TECH_NODES)}nm.', 32)

PERIPHERAL_PARAMS = {
    **ADDER_PARAMS, **SHIFT_ADD_PARAMS, **MAX_POOL_PARAMS,
    **ADDER_TREE_PARAMS, **MUX_PARAMS, **FLIP_FLOP_PARAMS
}
ALL_PARAMS = {
    **PIM_PARAMS, **ADDER_PARAMS, **SHIFT_ADD_PARAMS, **MAX_POOL_PARAMS,
    **ADDER_TREE_PARAMS, **MUX_PARAMS, **FLIP_FLOP_PARAMS
}

# Actions from Accelergy and their translations
READ_ACTIONS = [
    'read', 'mac', 'fire', 'drive', 'run', 'mac_random',
    'mac_reused', 'compute', 'add', 'shift_add', 'max_pool', 'maxpool'
    'convert', 'activate']
WRITE_ACTIONS = ['write', 'set', 'erase']
IDLE_ACTIONS = ['idle', 'gated_read',
                'gated_write', 'skipped_read', 'skipped_write']
ALL_ACTIONS = READ_ACTIONS + WRITE_ACTIONS + IDLE_ACTIONS

# Accelergy prmiitive components supported and their internal names
SUPPORTED_CLASSES = {
    'array_row_drivers':    (neurointerface.row_stats, PIM_PARAMS),
    'array_col_drivers':    (neurointerface.col_stats, PIM_PARAMS),
    'array_adc':            (neurointerface.col_stats, PIM_PARAMS),
    'memory_cell':          (neurointerface.cell_stats, PIM_PARAMS),
    'shift_add':            (neurointerface.shift_add_stats, SHIFT_ADD_PARAMS),
    'intadder':             (neurointerface.adder_stats, ADDER_PARAMS),
    'intadder_tree':        (neurointerface.adder_tree_stats, ADDER_TREE_PARAMS),
    'max_pool':             (neurointerface.max_pool_stats, MAX_POOL_PARAMS),
    'mux':                  (neurointerface.mux_stats, MUX_PARAMS),
    'flip_flop':            (neurointerface.flip_flop_stats, FLIP_FLOP_PARAMS),
    'not_gate':             (neurointerface.not_gate_stats, {}),
    'nand_gate':            (neurointerface.nand_gate_stats, {}),
    'nor_gate':             (neurointerface.nor_gate_stats, {}),
}

PRINTED_INTERP_WARNING = False

logger = None

# ==================================================================================================
# Input Parsing
# ==================================================================================================


def build_crossbar(attrs: dict) -> neurointerface.Crossbar:
    """ Builds a crossbar from the given attributes """
    cell_config = attrs['cell_config']
    peripheral_args = [(k, v)
                       for k, v in attrs.items() if k in PERIPHERAL_PARAMS]
    key = dict_to_str(attrs)
    attrs = {
        'sequential': attrs['sequential'],
        'rows': attrs['rows'],
        'cols': attrs['cols'],
        'cols_muxed': math.ceil(attrs['rows'] / attrs['cols_active_at_once']),
        'technology': attrs['technology'],
        'adc_resolution': attrs['adc_resolution'],
        'read_pulse_width': attrs['read_pulse_width'],
        'latency': attrs['latency'],
        'voltage_dac_bits': attrs['voltage_dac_bits'],
        'temporal_dac_bits': attrs['temporal_dac_bits'],
        'temporal_spiking': attrs['temporal_spiking'],
        'adc_action_share': attrs['adc_action_share'],
        'adc_area_share': attrs['adc_area_share'],
    }
    if key not in CACHE:
        CACHE[key] = neurointerface.Crossbar(**attrs)
        CACHE[key].run_neurosim(
            cell_config, neurointerface.DEFAULT_CONFIG, peripheral_args)
    else:
        logger.debug("Found cached output for %s. If you're looking for the "
                     "log for this, see previous debug messages.", key)
    return CACHE[key]


def get_neurosim_output(kind: str, attributes: dict) -> Dict[str, float]:
    """ Queries Neurosim for the stats for 'kind' component with 'attributes' attributes """
    assert kind in SUPPORTED_CLASSES, f'Unsupported primitive: {kind}'
    logger.debug(
        "Querying Neurosim for %s with attributes: %s", kind, attributes)

    # Load defaults
    to_pass = {k: v[1] for k, v in ALL_PARAMS.items()}
    # Get call function ready
    callfunc = SUPPORTED_CLASSES[kind][0]
    params = SUPPORTED_CLASSES[kind][1]
    docs = {k: v[0] for k, v in params.items()}

    # Get required parameters
    for p in params:
        if 'REQUIRED' in params[p][0]:
            assert p in attributes, f'Failed to generate {kind}. Required parameter not found: ' \
                                    f'{p}. Usage: \n{dict_to_str(docs)}'
        elif p not in attributes:
            attributes[p] = to_pass[p]

        passtype = params[p][2] if len(params[p]) > 2 else int
        try:
            if isinstance(attributes[p], str) and passtype != str:
                t = ''.join(c for c in attributes[p] if (
                    c.isdigit() or c == '.'))
            else:
                t = attributes[p]
            if t != attributes[p]:
                print(
                    f'WARN: Non-numeric {attributes[p]} for parameter {p}. Using {t} instead.')
            to_pass[p] = passtype(t)
        except ValueError as e:
            raise ValueError(f'Failed to generate {kind}. Parameter {p} must be of type '
                             f'{passtype}. Given: "{attributes[p]}" Usage: \n{dict_to_str(docs)}') from e

    tn = PERMITTED_TECH_NODES
    assert to_pass['rows'] >= 8, \
        f'Rows must be >=8. Got {to_pass["rows"]}'
    assert to_pass['cols'] >= 8, \
        f'Columns must be >=8. Given: {to_pass["columns"]}'
    assert to_pass['cols_active_at_once'] >= 1, \
        f'Columns active at once must be >=1 and divide evenly into cols. ' \
        f'Given: {to_pass["cols"]} cols, {to_pass["cols_active_at_once"]} cols active at once'
    assert min(tn) <= to_pass['technology'] <= max(tn), \
        f'Tech node must be between {max(tn)} and {min(tn)}nm. Given: {to_pass["technology"]}nm'
    assert to_pass['n_bits'] >= 1, \
        f'Adder resolution must be >=1. Given: {to_pass["n_bits"]}'
    assert to_pass['shift_register_n_bits'] >= 1, \
        f'Shift register resolution must be >=1. Given: {to_pass["shift_register_n_bits"]}'
    assert to_pass['pool_window'] >= 1, \
        f'Max pool window size must be >=1. Given: {to_pass["window_size"]}'
    assert to_pass['n_adder_tree_inputs'] > 0, \
        f'Number of adder tree inputs must be >=1. Given: {to_pass["n_adder_tree_inputs"]}'
    assert to_pass['n_mux_inputs'] > 0, \
        f'Number of mux inputs must be >=1. Given: {to_pass["n_mux_inputs"]}'
    assert to_pass['voltage_dac_bits'] > 0, \
        f'Voltage DAC bits must be >=1. Given: {to_pass["voltage_dac_bits"]}'
    assert to_pass['temporal_dac_bits'] > 0, \
        f'Temporal DAC bits must be >=1. Given: {to_pass["temporal_dac_bits"]}'

    if not os.path.exists(to_pass['cell_config']):
        cell_config = os.path.join(
            SCRIPT_DIR, 'cells', to_pass['cell_config'] + '.cell')
        assert os.path.exists(cell_config), f'Cell config {to_pass["cell_config"]}" not found. ' \
            f'Try a sample config: "{", ".join(SAMPLE_CELLS)}'
        to_pass['cell_config'] = cell_config

    # Interpolate the technology node. If p is in PERMITTED_TECH_NODES, then all this comes out
    # to just p. If p is not in PERMITTED_TECH_NODES, then we interpolate between the two closest.
    t = to_pass['technology']
    del to_pass['technology']
    hi = min(p for p in PERMITTED_TECH_NODES if p >= t)
    lo = max(p for p in PERMITTED_TECH_NODES if p <= t)
    interp_pt = (t - lo) / (hi - lo) if hi - lo else 0
    hi_crossbar = build_crossbar({**to_pass, 'technology': hi})
    lo_crossbar = build_crossbar({**to_pass, 'technology': lo})
    hi_est = callfunc(
        hi_crossbar, to_pass['average_input_value'], to_pass['average_cell_value'])
    lo_est = callfunc(
        lo_crossbar, to_pass['average_input_value'], to_pass['average_cell_value'])
    if hi != lo:
        logger.debug("Interpolating between %snm and %snm. Interpolation "
                     "point: %s", lo, hi, interp_pt)

    rval = {k: lo_est[k] + (hi_est[k] - lo_est[k]) * interp_pt for k in hi_est}
    logger.debug("NeuroSim returned: %s", rval)

    return rval


def query_neurosim(kind: str, attributes: dict) -> Dict[str, float]:
    for n in ['array_adc', 'array_col_drivers']:
        assert n in SUPPORTED_CLASSES, \
            'Please update this method body to support the new NeuroSim names.'

    if kind == 'array_col_drivers':
        attributes['adc_resolution'] = 0
        return get_neurosim_output(kind, attributes)

    if kind in ['array_adc', 'array_col_drivers']:
        logger.info('First running WITH the ADC to get total energy')
        with_adc = get_neurosim_output(kind, attributes)
        attributes['adc_resolution'] = 0
        logger.info('Now running WITHOUT the ADC to get column driver energy')
        without_adc = get_neurosim_output(kind, attributes)
        logger.info('Subtracting column driver energy to get ADC energy')
        return {k: with_adc[k] - without_adc[k] for k in with_adc}
    return get_neurosim_output(kind, attributes)


def dict_to_str(attributes: Dict) -> str:
    """ Converts a dictionary into a multi-line string representation """
    s = '\n'
    for k, v in attributes.items():
        s += f'\t{k}: {v}\n'
    return s

# ==============================================================================
# Wrapper Class
# ==============================================================================


class NeuroWrapper(AccelergyPlugIn):
    """ NVSIM wrapper class """

    def __init__(self):
        super().__init__()
        global logger  # pylint: disable=global-statement
        logger = self.logger
        neurointerface.logger = self.logger
        self.estimator_name = 'Neurosim Estimator'

    def primitive_action_supported(self, query: AccelergyQuery) -> AccuracyEstimation:
        class_name = query.class_name.lower()
        action_name = query.action_name.lower()
        if class_name in SUPPORTED_CLASSES:
            if action_name in ALL_ACTIONS:
                return AccuracyEstimation(ACCURACY)
            logger.error('ERROR: NeuroSim estimator supports %s but not '
                         'action %s. Supported actions are: %s',
                         class_name, action_name, ALL_ACTIONS)
            return AccuracyEstimation(0)
        logger.error(
            'ERROR: NeuroSim estimator does not support %s. Supported '
            'primitives are: %s', class_name, list(SUPPORTED_CLASSES.keys()))
        return AccuracyEstimation(0)

    def estimate_energy(self, query: AccelergyQuery) -> Estimation:
        class_name = query.class_name.lower()
        action_name = query.action_name.lower()
        attributes = query.class_attrs
        action_name = query.action_name.lower()
        stats = query_neurosim(class_name, attributes)
        assert action_name in ALL_ACTIONS, \
            f'{action_name} not supported. Must be in: {ALL_ACTIONS}'

        if action_name in READ_ACTIONS:
            v = stats['Read Energy']
        elif action_name in WRITE_ACTIONS:
            v = stats['Write Energy']
        else:
            v = stats['Leakage']
        return Estimation(v, 'p')

    def primitive_area_supported(self, query: AccelergyQuery) -> AccuracyEstimation:
        class_name = query.class_name.lower()
        if class_name.lower() in SUPPORTED_CLASSES:
            return AccuracyEstimation(ACCURACY)
        logger.error(
            'ERROR: NeuroSim estimator does not support %s. Supported '
            'primitives are: %s', class_name, list(SUPPORTED_CLASSES.keys()))
        acc = ACCURACY if class_name.lower() in SUPPORTED_CLASSES else 0
        return AccuracyEstimation(acc)

    def estimate_area(self, query: AccelergyQuery) -> Estimation:
        """
        :param interface:
        - contains two keys:
        1. class_name : string
        2. attributes: dictionary of name: value
        :type interface: dict
        :return the estimated area
        :rtype: float
        """
        a = query_neurosim(query.class_name.lower(), query.class_attrs)['Area']
        return Estimation(a, 'u^2')

    def get_name(self) -> str:
        """
        Returns the name of the plug-in.
        """
        return "Neurosim Plug-In"


if __name__ == '__main__':
    nw = NeuroWrapper()
    misc = {
        'class_name': 'mux',
        'action_name': 'read',
        'attributes': {'technology': 32, 'n_bits': 7, 'n_adder_tree_inputs': 1, 'n_mux_inputs': 32},
    }
    cols = {
        'class_name': 'array_col_drivers',
        'action_name': 'read',
        'attributes': {
            'technology': 32,
            'rows': 128,
            'cols': 128,
            'cols_active_at_once': 128,
            'cell_config': 'nvmexplorer_SRAM',
            'average_input_value': 1,
            'average_cell_value': 1,
            'sequential': 1,
            'adc_resolution': 0,
            'latency': 100},
    }

    a = []
    target = misc
    for i in [32]:
        target['attributes']['technology'] = i
        a.append(nw.estimate_energy(target))
        target['class_name'] = 'nand_gate'
        a.append(nw.estimate_area(target))
        target['class_name'] = 'not_gate'
        a.append(nw.estimate_area(target))
        target['class_name'] = 'nor_gate'
        a.append(nw.estimate_area(target))

    print('\n'.join(str(x) for x in a))

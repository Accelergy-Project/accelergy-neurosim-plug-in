""" Neurosim Accelergy Wrapper """
import math
import sys
import os
from typing import Dict
# Need to add this directory to path for proper imports
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(SCRIPT_DIR)
import neurointerface

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
    'technology':               ('REQUIRED: Technology node. Must be between 7 and 130nm.', 32),
    'rows' :                    ('REQUIRED: Number of rows in a crossbar.', 32),
    'cols':                     ('REQUIRED: Number of columns in a crossbar.', 32),
    'cols_active_at_once':      ('REQUIRED: Number of columns active at once.', 8),
    'cell_config':              (f'REQUIRED: Path to a NVSim cell config file to use, or one of '
                                 f'the following samples: ' +
                                 f', '.join(f'"{s}"' for s in SAMPLE_CELLS), 'placeholder',
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
}

ADDER_PARAMS = {
    'precision':                (f'REQUIRED: # Bits of the adder.', 8),
}
SHIFT_ADD_PARAMS = {
    'technology':               (f'REQUIRED: Technology node. Must be between 7 and 130nm.', 32),
    'precision':                (f'REQUIRED: # Bits of the adder.', 8),
    'shift_register_precision': (f'REQUIRED: # Bits of the shift register.', 16),
}
MAX_POOL_PARAMS = {
    'technology':               (f'REQUIRED: Technology node. Must be between 7 and 130nm.', 32),
    'precision':                (f'REQUIRED: # Bits.', 8),
    'pool_window':              (f'REQUIRED: Window size of max pooling.', 2),
}
ADDER_TREE_PARAMS = {
    'precision':                (f'REQUIRED: # Bits of the leaf adder.', 8),
    'n_adder_tree_inputs':      (f'REQUIRED: Number of inputs to the adder tree.', 2),
}

PARAM_DICTS = [PIM_PARAMS, ADDER_PARAMS, SHIFT_ADD_PARAMS, MAX_POOL_PARAMS, ADDER_TREE_PARAMS]
for p in PARAM_DICTS:
    p['technology'] = (f'REQUIRED: Technology node. Must be between {max(PERMITTED_TECH_NODES)}  '
                       f'and {min(PERMITTED_TECH_NODES)}nm.', 32)

PERIPHERAL_PARAMS = {**ADDER_PARAMS, **SHIFT_ADD_PARAMS, **MAX_POOL_PARAMS, **ADDER_TREE_PARAMS}
ALL_PARAMS = {**PIM_PARAMS, **ADDER_PARAMS, **SHIFT_ADD_PARAMS, **MAX_POOL_PARAMS,
              **ADDER_TREE_PARAMS}

# Actions from Accelergy and their translations
READ_ACTIONS = [
    'read', 'mac', 'fire', 'drive', 'run', 'mac_random',
    'mac_reused', 'compute', 'add', 'shift_add', 'max_pool', 'maxpool']
WRITE_ACTIONS = ['write', 'set', 'erase']
IDLE_ACTIONS = ['idle', 'gated_read', 'gated_write', 'skipped_read', 'skipped_write']
ALL_ACTIONS = READ_ACTIONS + WRITE_ACTIONS + IDLE_ACTIONS

# Accelergy prmiitive components supported and their internal names
SUPPORTED_CLASSES = {
    'pim_row_drivers' :     (neurointerface.row_stats, PIM_PARAMS),
    'pim_col_drivers' :     (neurointerface.col_stats, PIM_PARAMS),
    'pim_cell' :            (neurointerface.cell_stats, PIM_PARAMS),
    'shift_add':            (neurointerface.shift_add_stats, SHIFT_ADD_PARAMS),
    'neurosim_adder':       (neurointerface.adder_stats, ADDER_PARAMS),
    'neurosim_adder_tree':  (neurointerface.adder_tree_stats, ADDER_TREE_PARAMS),
    'max_pool':             (neurointerface.max_pool_stats, MAX_POOL_PARAMS),
}

PRINTED_INTERP_WARNING = False


# ==================================================================================================
# Input Parsing
# ==================================================================================================
def build_crossbar(attrs: dict) -> neurointerface.Crossbar:
    """ Builds a crossbar from the given attributes """
    cell_config = attrs['cell_config']
    peripheral_args = [(k, v) for k, v in attrs.items() if k in PERIPHERAL_PARAMS]
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
    }
    if key not in CACHE:
        CACHE[key] = neurointerface.Crossbar(**attrs)
        CACHE[key].run_neurosim(cell_config, neurointerface.DEFAULT_CONFIG, peripheral_args)
    return CACHE[key]


def query_neurosim(kind: str, attributes: dict) -> Dict[str, float]:
    """ Queries Neurosim for the stats for 'kind' component with 'attributes' attributes """
    assert kind in SUPPORTED_CLASSES, f'Unsupported primitive: {kind}'
    if DEBUG:
        print(f'Info: Querying Neurosim for {kind} with attributes: {attributes}')

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
                t = ''.join(c for c in attributes[p] if (c.isdigit() or c == '.'))
            else:
                t = attributes[p]
            if t != attributes[p]:
                print(f'WARN: Non-numeric {attributes[p]} for parameter {p}. Using {t} instead.')
            to_pass[p] = passtype(t)
        except ValueError:
            raise ValueError(f'Failed to generate {kind}. Parameter {p} must be of type ' \
                             f'{passtype}. Given: "{attributes[p]}" Usage: \n{dict_to_str(docs)}')

    tn = PERMITTED_TECH_NODES
    assert to_pass['rows'] > 8, \
        f'Rows must be >=8. Got {to_pass["rows"]}'
    assert to_pass['cols'] > 8, \
        f'Columns must be >=8. Given: {to_pass["columns"]}'
    assert to_pass['cols_active_at_once'] >= 1, \
        f'Columns active at once must be >=1 and divide evenly into cols. ' \
        f'Given: {to_pass["cols"]} cols, {to_pass["cols_active_at_once"]} cols active at once'
    assert min(tn) <= to_pass['technology'] <= max(tn), \
        f'Tech node must be between {max(tn)} and {min(tn)}nm. Given: {to_pass["technology"]}nm'
    assert to_pass['precision'] > 1, \
        f'Adder resolution must be >=1. Given: {to_pass["precision"]}'
    assert to_pass['shift_register_precision'] > 1, \
        f'Shift register resolution must be >=1. Given: {to_pass["shift_register_precision"]}'
    assert to_pass['pool_window'] > 1, \
        f'Max pool window size must be >=1. Given: {to_pass["window_size"]}'
    assert to_pass['n_adder_tree_inputs'] > 0, \
        f'Number of adder tree inputs must be >=1. Given: {to_pass["n_adder_tree_inputs"]}'

    if not os.path.exists(to_pass['cell_config']):
        cell_config = os.path.join(SCRIPT_DIR, 'cells', to_pass['cell_config'] + '.cell')
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
    hi_est = callfunc(hi_crossbar, to_pass['average_input_value'], to_pass['average_cell_value'])
    lo_est = callfunc(lo_crossbar, to_pass['average_input_value'], to_pass['average_cell_value'])
    if hi != lo:
        print(f'Info: Interpolating between {lo}nm and {hi}nm. Interpolation point: {interp_pt}')

    rval = {k: lo_est[k] + (hi_est[k] - lo_est[k]) * interp_pt for k in hi_est}
    if DEBUG:
        print(f'Info: NeuroSim returned: {rval}')

    return rval



def dict_to_str(attributes: Dict) -> str:
    """ Converts a dictionary into a multi-line string representation """
    s = '\n'
    for k, v in attributes.items():
        s += f'\t{k}: {v}\n'
    return s

# ==============================================================================
# Wrapper Class
# ==============================================================================
class NeuroWrapper:
    """ NVSIM wrapper class """
    def __init__(self):
        self.estimator_name = 'Neurosim Estimator'

    def primitive_action_supported(self, interface):
        """
        :param interface:
        - contains four keys:
        1. class_name : string
        2. attributes: dictionary of name: value
        3. action_name: string
        4. arguments: dictionary of name: value
        :type interface: dict
        :return return the accuracy if supported, return 0 if not
        :rtype: int
        """
        class_name = str(interface['class_name']).lower()
        action_name = str(interface['action_name']).lower()
        if class_name in SUPPORTED_CLASSES:
            if action_name in ALL_ACTIONS:
                return ACCURACY
            print(f'ERROR: NeuroSim estimator supports {class_name} but not action {action_name}')
            print(f'ERROR: Supported actions: {ALL_ACTIONS}')
        return 0


    def estimate_energy(self, interface):
        """
        :param interface:
        - contains four keys:
        1. class_name : string
        2. attributes: dictionary of name: value
        3. action_name: string
        4. arguments: dictionary of name: value
        :type interface: dict
        :return return the accuracy if supported, return 0 if not
        :rtype: int
        """
        class_name = str(interface['class_name']).lower()
        attributes = interface['attributes']
        action_name = str(interface['action_name']).lower()
        stats = query_neurosim(class_name, attributes)
        assert action_name in ALL_ACTIONS, f'{action_name} not supported. Must be in: {ALL_ACTIONS}'

        if action_name in READ_ACTIONS:
            return stats['Read Energy']
        if action_name in WRITE_ACTIONS:
            return stats['Write Energy']
        return stats['Leakage']

    def primitive_area_supported(self, interface):
        """
        :param interface:
        - contains two keys:
        1. class_name : string
        2. attributes: dictionary of name: value
        :type interface: dict
        :return return the accuracy if supported, return 0 if not
        :rtype: int
        """
        return ACCURACY if interface['class_name'].lower() in SUPPORTED_CLASSES else 0

    def estimate_area(self, interface):
        """
        :param interface:
        - contains two keys:
        1. class_name : string
        2. attributes: dictionary of name: value
        :type interface: dict
        :return the estimated area
        :rtype: float
        """
        class_name = interface['class_name'].lower()
        attributes = interface['attributes']

        return query_neurosim(class_name, attributes)['Area']


# TEST CODE


PIM_PARAMS = {
    'technology':               ('REQUIRED: Technology node. Must be between 7 and 130nm.', 32),
    'rows' :                    ('REQUIRED: Number of rows in a crossbar.', 32),
    'cols':                     ('REQUIRED: Number of columns in a crossbar.', 32),
    'cols_active_at_once':      ('REQUIRED: Number of columns active at once.', 8),
    'cell_config':              (f'REQUIRED: Path to a NVSim cell config file to use, or one of '
                                 f'the following samples: ' +
                                 f', '.join(f'"{s}"' for s in SAMPLE_CELLS), 'placeholder',
                                 str),
    'average_input_value':      (f' REQUIRED: Average input value to a row. Must be between '
                                 f'0 and 1.', 1, float),
    'average_cell_value':       (f' REQUIRED: Average cell value. Must be between 0 and 1.', 1,
                                 float),
    'sequential':               (f'OPTIONAL: Sequential mode. Default is False. If True, the '
                                 f'crossbar will be set up to activate one row at a time. '
                                 f'Can be used as a memory this way.', False),
    'adc_resolution':           (f'OPTIONAL: ADC resolution. Set this if to use Neurosim\'s '
                                 f'build-in ADC. Default is False.', 0)
}


if __name__ == '__main__':
    nw = NeuroWrapper()
    misc = {
        'class_name': 'neurosim_adder_tree',
        'action_name': 'read',
        'attributes': {'technology': 32, 'precision': 32, 'n_adder_tree_inputs': 1},
    }
    cols = {
        'class_name': 'pim_col_drivers',
        'action_name': 'read',
        'attributes': {
            'technology': 32,
            'rows': 128,
            'cols': 128,
            'cols_active_at_once': 256,
            'cell_config': 'nvmexplorer_RRAM',
            'average_input_value': 1,
            'average_cell_value': 1,
            'sequential': 1,
            'adc_resolution': 0},
    }

    a = []
    #for i in [32, 45]:
    #    x['attributes']['technology'] = i
    #    a.append(nw.estimate_energy(x))

    #for adc_resolution in range(1, 12):
    #    cols['attributes']['adc_resolution'] = adc_resolution
    #    a.append(nw.estimate_energy(cols))
    import copy
    for i in range(1, 128):
        misc['attributes']['n_adder_tree_inputs'] = i
        misc2 = copy.deepcopy(misc)
        misc2['action_name'] = 'idle'
        a.append(nw.estimate_energy(misc2))

    print('\n'.join(str(x) for x in a))

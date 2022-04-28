import os
import neurointerface

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

results = {}
for c in os.listdir('./cells'):
    print(c)
    extras = [(k, v[1]) for k, v in {**ADDER_PARAMS, **SHIFT_ADD_PARAMS, **MAX_POOL_PARAMS, **ADDER_TREE_PARAMS}.items()]
    xbar = neurointerface.Crossbar(0, 128, 128, 128, 32, 0, 1e-8)
    xbar.run_neurosim('./cells/' + c, neurointerface.DEFAULT_CONFIG, extras)
    print(neurointerface.cell_stats(xbar, 0.5, 0.5))
    results[c] = neurointerface.cell_stats(xbar, 0.5, 0.5)
    
for k, v in results.items():
    print(f'{k}: {v}')

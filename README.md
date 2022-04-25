# NeuroSim Accelergy Plugin
This Accelery plugin provides estimations for Neurosim components. It enables estimation for analog Processing-In-Memory Crossbar components and various peripheral components.

We also allow conversions for NVSim/NVMExplorer style cell files to Neurosim crossbars.

## PIM Crossbar Model
We encourage you read section 6 of the Neurosim Manual in the DNN_NeuroSim_V1.3/Documents folder.

PIM crossbar energy/area is dominated by four pieces.

- Row Activation: This is the path for inputs to activate a crossbar. An input voltage will actiavte a wordline, or row of memory cells. Multiple input voltages or longer-duration inputs can feed higher resolution inputs.
- Column Activation: This is the path for outputs to be read out of a crossbar. Once input voltages have activated a wordline, they will feed current onto bitlines, or columns. Column activation is the energy required to get this current out and to precharge the wordline before reading.
- Cell: This a memory cells in the crossbar.
- Analog-Digital-Conversion: This is the circuit that reads an analog value from the crossbar and converts to a digital signal for output processing. 

In short, an input activates a row, an output activates a column, a multiplication activates a cell, and an output read takes an ADC conversion.

## Installing
```
git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-neurosim-plugin.git
cd accelergy-neurosim-plugin
python3 setup.py build_ext && pip install .
```

## Creating Custom Cells
Cell files must follow the NVSim/NVMExplorer format. See the cells/ directory for references. 
PIM crossbars can be created with any user-defined cell.

## PIM Components
We support four components for estimating PIM crossbar energy. They take the following parameters.

### Arguments
- technology: The technology node in nm
- row: The number of rows in the crossbar
- cols: The number of columns in the crossbar
- cols_active_at_once: The number of columns to be activated at a single time
- cell_config: The path to a NVSim/NVMExplorer cell file, or a sample cell to use.
- average_input_value: A value between 0 and 1 reflecting the average input being sent on rows. For example, if inputs are encoded as values from 0-4 with an average of 3, average_input_value is 0.75.
- average_cell_value: Like average_input_value, but with encoded weights. For example, if weights are encoded as values from 1-10 with an average of 7, average_cell_value is 0.7.
- sequential: A binary value. If true, rows are addressed and accessed one at a time. Otherwise, rows are to be activated in large blocks and not addressed. Setting this to TRUE can simulate a PIM memory.
- adc_resolution: This is the number of bits of the ADC used for readout. ADC is a flash ADC. To exclude the ADC and use your own, set adc_resolution to 0.

## Peripheral Components
We support other digital components from NeuroSim. These components are useful in many places-- not just PIM!

### Integer Adder
Yep, it adds integers.

Class:
- neurosim_adder

Required parameters:
- technology: Technology node in nm
- precision: Number of bits to add

Actions:
- add

### Integer Shift-Add
An adder + a shift register to accumulate variable-precision values.

Class:
- shift_add

Required parameters:
- technology: Technology node in nm
- precision: Number of bits to add
- shift_register_precision: Number of bits in the shift register

Actions:
- shift_add

### Integer Adder-Tree
A tree of adders to accmulate many values. Each level of the tree adds with higher precision to
ensure no overflow.

Class:
- neurosim_adder_tree

Required parameters:
- technology: Technology node in nm
- precision: Number of bits of a leaf adder. This is the minimum precision in the tree. Each additional level will add a bit.
- n_adder_tree_inputs: Number of values to add

Actions:
- add

### Integer Max-Pool
A max-pooling unit that finds and outputs the maximum a set of values.

Class:
- max_pool

Required parameters:
- technology: Technology node in nm
- precision: Number of bits for each input value
- pool_window: Number of values to compare

Actions:
- max_pool

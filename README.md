# NeuroSim Accelergy Plugin
This Accelery plugin provides estimations for NeuroSim[1] components. It
enables estimation for analog Processing-In-Memory (PIM) Array components and
various peripheral components.

We also allow conversions for NVMExplorer[2]/NVSim[3] style cell files to
Neurosim processing-in-memory / compute-in-memory arrays.

## PIM Array Model
We encourage you read section 6 of the NeuroSim Manual in the
DNN_NeuroSim_V1.3/Documents folder.

We model PIM array energy/area with four pieces:

- Row Activation: This is the path for inputs to activate a array. An input
  voltage will actiavte a wordline, or row of memory cells. Multiple input
  voltages or longer-duration inputs can feed higher resolution inputs.
- Column Activation: This is the path for outputs to be read out of a array.
  Once input voltages have activated a wordline, they will feed current onto
  bitlines, or columns. Column activation is the energy required to get this
  current out and to precharge the wordline before reading.
- Analog-Digital-Conversion: This is the circuit that reads an analog value
  from the array and converts to a digital signal for output processing. 
- Cell: This a memory cells in the array.

In short: An input activates a row, an output activates a column, a
multiplication activates a cell, and an output read takes an ADC conversion.

## Installing
```
git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-neurosim-plugin.git
cd accelergy-neurosim-plugin
python3 setup.py build_ext && pip install .
```

## Creating Custom Cells
Cell files must follow the NVMExplorer[2]/NVSim[3] format. See the cells/
directory for references. PIM arrays can be created with any user-defined
cell.

# PIM Components
We support four components for estimating PIM array energy.

- array_row_drivers: These activate rows.
- array_col_drivers: These activate columns.
- array_adc: These read out analog values into digital signals.
- memory_cell: This is a cell in a array.

## PIM Component Arguments
PIM components take the following arguments.
- technology: The technology node in nm
- row: The number of rows in the array
- cols: The number of columns in the array
- cols_active_at_once: The number of columns to be activated at a single time
- cell_config: The path to a NVMExplorer[2]/NVSim[3] cell file, or a sample
  cell to use.
- average_input_value: A value between 0 and 1 reflecting the average input
  being sent on rows. For example, if inputs are encoded as values from 0-4
  with an average of 3, average_input_value is 0.75.
- average_cell_value: Like average_input_value, but with encoded weights. For
  example, if weights are encoded as values from 1-10 with an average of 7,
  average_cell_value is 0.7.
- sequential: A binary value. If true, rows are addressed and accessed one at a
  time. Otherwise, rows are to be activated in large blocks and not addressed.
  Setting this to TRUE can simulate a PIM memory.
- adc_resolution: This is the number of bits of the ADC used for readout. ADC
  is a flash ADC. To exclude the ADC and use your own, set adc_resolution to 0.
- read_pulse_width: Number of ns each read pulse lasts.
- voltage_dac_bits: Number of bits resolution for a voltage-based DAC on each
  row. Voltage DACs use a row switch connected to a power rail for each
  possible input value.
- temporal_dac_bits: Number of bits resolution for a temporal DAC on each row.
  Temporal DACs enode inputs as an amount of time the row stays high.

# Peripheral Components
We support other digital components from NeuroSim. These components are useful
in many places-- not just PIM!

## Integer Adder
Yep, it adds integers.

Class:
- intadder

Required parameters:
- technology: Technology node in nm
- n_bits: Number of bits to add

Actions:
- add

## Integer Shift-Add
An adder + a shift register to accumulate variable-precision values.

Class:
- shift_add

Required parameters:
- technology: Technology node in nm
- n_bits: Number of bits to add
- shift_register_n_bits: Number of bits in the shift register

Actions:
- shift_add

## Integer Adder-Tree
A tree of adders to accmulate many values. Each level of the tree adds with
higher n_bits to ensure no overflow.

Class:
- intadder_tree

Required parameters:
- technology: Technology node in nm
- n_bits: Number of bits of a leaf adder. This is the minimum n_bits in the
  tree. Each additional level will add a bit.
- n_adder_tree_inputs: Number of values to add

Actions:
- add

## Integer Max-Pool
A max-pooling unit that finds and outputs the maximum a set of values.

Class:
- max_pool

Required parameters:
- technology: Technology node in nm
- n_bits: Number of bits for each input value
- pool_window: Number of values to compare

Actions:
- max_pool

## Multiplexer
An n-bit multiplexer

Class:
- mux

Required parameters:
- technology: Technology node in nm
- n_bits: Number of bits for each input value
- n_inputs: Number of inputs to the mux

Actions:
- max_pool

## Flip-flop
A digital flip flop

Class:
- flip_flop

Required parameters:
- technology: Technology node in nm
- n_bits: Number of flip flop bits

Actions:
- max_pool

## NOT, NAND, and NOR gates
Logic gates

Class:
- not_gate
- nand_gate
- nor_gate

Required parameters:
- technology: Technology node in nm

Actions:
- read

### References
[1]X. Peng, S. Huang, Y. Luo, X. Sun, and S. Yu, “DNN+NeuroSim: An End-to-End
Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device
Technologies,” in 2019 IEEE International Electron Devices Meeting (IEDM), Dec.
2019, p. 32.5.1-32.5.4. doi: 10.1109/IEDM19573.2019.8993491.

[2]L. Pentecost, A. Hankin, M. Donato, M. Hempstead, G.-Y. Wei, and D. Brooks,
NVMExplorer: A Framework for Cross-Stack Comparisons of Embedded Non-Volatile
Memories. 2021.

[3]X. Dong, C. Xu, Y. Xie, and N. P. Jouppi, “NVSim: A Circuit-Level
Performance, Energy, and Area Model for Emerging Nonvolatile Memory,” IEEE
Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol.
31, no. 7, pp. 994–1007, Jul. 2012, doi: 10.1109/TCAD.2012.2185930.

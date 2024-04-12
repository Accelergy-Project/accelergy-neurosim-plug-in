"""
Interface for Neurosim. Exposes row, column, and cell energy in terms of OFF and ON actions. Also
includes integration of cell files from NVSim and NVMExplorer.
"""

from statistics import mean
import threading
from typing import Dict, List, Tuple
from numbers import Number
import os
import subprocess
import re
import os

MY_PID = os.getpid()
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "default_config.cfg")
NEUROSIM_PATH = os.path.join(SCRIPT_DIR, "NeuroSim/main")
CFG_WRITE_PATH = os.path.join(SCRIPT_DIR, f"./neurosim_input_{MY_PID}.cfg")

logger = None

# ==================================================================================================
# NVSIM/NVMEXPLORER -> NEUROSIM TRANSLATIONS
# ==================================================================================================

PARSED = {}

# These should all be found in the NVSim config file
NV_TO_NEURO_TERNARIES = {
    # SRAM if -MemCellType: SRAM is present, else RRAM
    "-MemCellType: SRAM": ("memcelltype:", 1, 2),
    "-AccessType: CMOS": ("accesstype:", 1, 4),
    "-ReadMode: current": ("currentMode:", 1, 0),
    # Not used in NeuroSim but here so we can
    "-SetMode: current": ("currentModeSET:", 1, 0),
    # calculate the SET voltage
}

# These are parsed in order, so make sure all lambdas have required data first.
# Formats:
# Standard: NVSim format : NeuroSim format
# Scaled: NVSim format : (NeuroSim format, Scale factor)
# Lambda: Neurosim format : lambda
# Options: (NVsim format 1, NVsim format 2..) : NeuroSim format
NV_TO_NEURO = [
    ##
    # Device agnostic
    ("-CellArea (F^2):", "CELL_AREA"),  # Get these into the value cache
    ("-CellAspectRatio:", "CELL_ASPECT_RATIO"),
    (
        "heightInFeatureSize1T1R:",
        lambda: (PARSED["CELL_AREA"] * PARSED["CELL_ASPECT_RATIO"]) ** 0.5,
    ),
    (
        "widthInFeatureSize1T1R:",
        lambda: (PARSED["CELL_AREA"] / PARSED["CELL_ASPECT_RATIO"]) ** 0.5,
    ),
    ("heightInFeatureSizeCrossbar:", lambda: PARSED["heightInFeatureSize1T1R:"]),
    ("widthInFeatureSizeCrossbar:", lambda: PARSED["widthInFeatureSize1T1R:"]),
    ("heightInFeatureSizeSRAM:", lambda: PARSED["heightInFeatureSize1T1R:"]),
    ("widthInFeatureSizeSRAM:", lambda: PARSED["widthInFeatureSize1T1R:"]),
    ##
    # SRAM Configuration
    ("-AccessCMOSWidth (F):", "widthAccessCMOS:"),
    ("-SRAMCellNMOSWidth (F):", "widthSRAMCellNMOS:"),
    ("-SRAMCellPMOSWidth (F):", "widthSRAMCellPMOS:"),
    ("-MinSenseVoltage (mV):", ("minSenseVoltage:", 1e-3)),
    ##
    # RRAM Configuration
    (("-ResistanceOn (ohm):", "-ResistanceOnAtReadVoltage (ohm):"), "resistanceOn:"),
    (("-ResistanceOff (ohm):", "-ResistanceOffAtReadVoltage (ohm):"), "resistanceOff:"),
    ("-AccessTransistorResistance (ohm):", "accessTransistorResistance:"),
    # If the read/write mode is current, set the voltage to the current * avg resistance
    ("AVG_RES", lambda: (PARSED["resistanceOn:"] + PARSED["resistanceOff:"]) / 2),
    (
        "AVG_CONDUCTANCE",
        lambda: (1 / PARSED["resistanceOn:"] + 1 / PARSED["resistanceOff:"]) / 2,
    ),
    ("-SetVoltage (V):", "writeVoltage:"),
    ("-WriteVoltage (V):", "WriteVoltage:"),
    ("-SetPulse (ns):", ("writePulseWidth:", 1e-9)),
    ("-SetEnergy (pJ):", ("SET_ENERGY", 1e-12)),
    ("-SetCurrent (uA):", ("SET_CURRENT", 1e-6)),
    ("-SetPower (uW):", ("SET_POWER", 1e-6)),
    (
        "writeVoltage:",
        lambda: PARSED.get(
            "writeVoltage:", PARSED["SET_POWER"] / PARSED["AVG_CONDUCTANCE"] ** 2
        ),
    ),
    (
        "writeVoltage:",
        lambda: PARSED.get(
            "writeVoltage:", PARSED["SET_CURRENT"] / PARSED["AVG_CONDUCTANCE"]
        ),
    ),
    (
        "writePulseWidth:",
        lambda: PARSED["SET_ENERGY"]
        / (PARSED["writeVoltage:"] ** 2 * PARSED["AVG_CONDUCTANCE"]),
    ),
    ("-ReadVoltage (V):", "readVoltage:"),
    # ('-ReadPulse (ns):', ('readPulseWidth:', 1e-9)),
    ("-ReadCurrent (uA):", ("READ_CURRENT", 1e-6)),
    ("-ReadEnergy (pJ):", ("READ_ENERGY", 1e-12)),
    ("-ReadPower (uW):", ("READ_POWER", 1e-6)),
    (
        "readVoltage:",
        lambda: PARSED.get(
            "readVoltage:", PARSED["READ_POWER"] / PARSED["AVG_CONDUCTANCE"] ** 2
        ),
    ),
    (
        "readVoltage:",
        lambda: PARSED.get(
            "readVoltage:", PARSED["READ_CURRENT"] / PARSED["AVG_CONDUCTANCE"]
        ),
    ),
    # ('readPulseWidth:', lambda: PARSED['READ_ENERGY'] \
    #    / (PARSED['readVoltage:'] ** 2 * PARSED['AVG_CONDUCTANCE'])),
    # ('readPulseWidth:', lambda: PARSED.get('readPulseWidth:', 1e-8)),
    ("-AccessVoltage (V):", "accessVoltage:"),
    ("accessVoltage:", lambda: PARSED.get("accessVoltage:", PARSED["readVoltage:"])),
    ("-CellCapacitanceMultiplier:", "cellCapacitanceMultiplier:"),
    ("-CellCapacitanceAdjust (F):", "cellCapacitanceAdjust:"),
    ("-DeviceRoadmap -1LP 1HP 2LSTP:", "deviceRoadmap:"),
]


# ==================================================================================================
# NEUROSIM AND NVSIM CONFIG FILE PARSING. THESE FUNCTIONS ARE USED BEFORE NEUROSIM IS CALLED.
# ==================================================================================================


def grabstat(
    prefixes: str, text: str, isstr: bool = False, default: None = None
) -> float or str:
    """
    Grabs the first item after all prefixes in text. Stat should be prefixed with = or :. Only grabs
    floats unless isstr is true. Prefixes should be a comma-separated list with no spaces.
    """
    prefixes = prefixes.split(",")
    org = text
    for pre in prefixes:
        text = text.split(pre, 1)
        if len(text) != 2 and default is not None:
            return default
        assert len(text) == 2, (
            f"Could not find {pre} in text. Text below:\n"
            + org.replace("\n", "\t\t\n")
            + f"\nCould not find {pre} in text. Text above."
        )
        text = text[1]
    if isstr:
        return re.search(r"[:=]\s*([^\n]+)", text).group(1)
    return float(re.search(r"[:=]\s*(\d+\.\d+|\d+)", text).group(1))


def nvsimget(getting: str, text: str, check_if_present: bool) -> float or bool:
    """
    Checks for the presence of 'getting' starting a line in text.
    If check_if_present: Return true/false whether getting is present
    Otherwise: Return the following double. Error on invalid double
        If presence true, return True
        If presence false, return the next value. Must be a valid float
    """
    getting = re.escape(getting)
    found = re.search(rf"^{getting}", text, flags=re.MULTILINE)
    if check_if_present:
        return found is not None

    found = re.search(rf"^{getting}\s*([a-zA-Z0-9\.\-]+)", text, flags=re.MULTILINE)
    try:
        return float(found.group(1))
    except:
        raise Exception(
            f"Could not find valid float setting for {getting} in cell file."
        )


def cfgset(setting: str, text: str, value: float):
    """Sets 'setting' in the config file to 'value'"""
    setting = re.escape(setting)
    text = re.sub(
        rf"^({setting}\s*)([a-zA-Z0-9\.\-]+)",
        rf"\g<1>{str(value)}",
        text,
        flags=re.MULTILINE,
    )
    return text


def buildcfg(cellpath: str, cfgpath: str) -> Tuple[str, Dict[str, float]]:
    """Populates a Neurosim config with the values from the cell file and returns contents"""
    PARSED.clear()
    other_vars = {}
    fails = {}
    with open(cellpath, "r") as f:
        celltext = f.read()
    with open(cfgpath, "r") as f:
        cfgtext = f.read()

    # Parse ternaries
    for k, v in NV_TO_NEURO_TERNARIES.items():
        PARSED[v[0]] = v[1] if nvsimget(k, celltext, True) else v[2]

    if nvsimget("-CellReadLeakEnergyMultiplier:", celltext, True):
        other_vars["cell_read_leak_energy_mult"] = nvsimget(
            "-CellReadLeakEnergyMultiplier:", celltext, False
        )
    if nvsimget("-CellWriteEnergyMultiplier:", celltext, True):
        other_vars["cell_write_energy_mult"] = nvsimget(
            "-CellWriteEnergyMultiplier:", celltext, False
        )
    if nvsimget("-CellCapacitanceMultiplier:", celltext, True):
        other_vars["cell_capacitance_mult"] = nvsimget(
            "-CellCapacitanceMultiplier:", celltext, False
        )

    logger.info("Neurosim Plugin parsing cell file...")

    # Parse expressions
    for n in NV_TO_NEURO:
        key, v = n
        errors = []
        if not isinstance(key, tuple):
            key = (key,)
        for k in key:
            # Parse scale factor
            neuroname, scale = v if isinstance(v, tuple) else (v, 1)

            # If it's callable, call it
            if callable(v):
                neuroname = k
                try:
                    PARSED[neuroname] = v()
                    logger.debug("\t%s=%s", neuroname, PARSED[neuroname])
                    break
                except KeyError as e:
                    errors.append(f"Could not find value of {e}.")
                except Exception as e:
                    errors.append(f"{type(e)}({e}).")
            # If it's not callable, try to grab it from the cell file
            elif nvsimget(k, celltext, True):
                PARSED[neuroname] = nvsimget(k, celltext, False) * scale
                logger.debug("\t%s=%s", neuroname, PARSED[neuroname])
                break
            else:
                errors.append(f"Could not find {k} in cell file.")
        # If we failed, record errors
        else:
            failstr = f'\tFailed to calculate "{neuroname}". Ignore if this value is not needed.'
            fails[neuroname] = fails.get(neuroname, [failstr]) + [
                f"\t\t{e}" for e in errors
            ]

    for k, v in fails.items():
        if k in PARSED:  # It succeeded somewhere else!
            continue
        for f in v:
            logger.info(f)
    for k, v in PARSED.items():
        cfgtext = cfgset(k, cfgtext, v)

    return cfgtext, other_vars


# ==================================================================================================
# NEUROSIM OUTPUT PARSING. THESE FUNCTIONS ARE USED AFTER NEUROSIM IS CALLED.
# ==================================================================================================


class Component:
    """Class to store a single Neurosim component"""

    def __init__(self, line: str):
        line = [s.strip().lower() for s in line.split(",")]
        self.read = "read" in line.pop(0)
        self.name = line.pop(0)

        def x():
            return float(line.pop(0))

        self.activation_energy, self.energy_per_row, self.energy_per_col = x(), x(), x()
        self.energy_per_cell, self.area, self.leakage = x(), x(), x()


def replace_cfg(find: str, replace: str or Number, text: str, cfgfile: str):
    """Replaces 'find' in text with 'replace'. Errors on not found."""
    find = f"SETME_{find.upper()}"
    if find not in text:
        logger.error("OFFENDING CONFIG FILE BELOW.")
        logger.error("|\t" + text.replace("\n", "\n| "))
        logger.error("OFFENDING CONFIG FILE ABOVE.")
        raise ValueError(
            f"{find} not found in {cfgfile}. Is the default config file altered?"
        )
    return text.replace(find, str(replace))


class Crossbar:
    """Holds one crossbar and related peripherals."""

    def __init__(
        self,
        sequential: bool,
        rows: int,
        cols: int,
        cols_muxed: int,
        technology: int,
        adc_resolution: int,
        read_pulse_width: float,
        global_cycle_seconds: float,
        cycle_seconds: float,
        voltage_dac_bits: int,
        temporal_dac_bits: int,
        temporal_spiking: bool,
        voltage: float,
        threshold_voltage: float,
    ):

        self.comps = []
        self.sequential = 1 if sequential else 2
        self.rows = rows
        self.cols = cols
        self.cols_muxed = cols_muxed
        self.technology = technology
        self.num_output_levels = 2**adc_resolution
        self.has_adc = adc_resolution > 0
        self.read_pulse_width = read_pulse_width
        self.global_cycle_seconds = global_cycle_seconds
        self.cycle_seconds = cycle_seconds
        self.voltage_dac_bits = voltage_dac_bits
        self.temporal_dac_bits = temporal_dac_bits
        self.temporal_spiking = temporal_spiking
        self.adc_energy_scale = 1
        self.adc_area_scale = 1
        self.cell_read_leak_energy_scale = 1
        self.cell_write_energy_scale = 1
        self.voltage = voltage
        self.threshold_voltage = threshold_voltage

        self.max_activation_time = 2**self.temporal_dac_bits - 1

    def run_neurosim(
        self, cellfile: str, cfgfile: str, other_args: List[Tuple[str, Number]] = ()
    ):
        """Runs Neurosim with the given parameters. Populates component data from the output."""
        logger.info("Building a crossbar with cell file %s", cellfile)

        # Build config
        cfg, other_vars = buildcfg(cellfile, cfgfile)
        self.cell_read_leak_energy_scale *= other_vars.get(
            "cell_read_leak_energy_mult", 1
        )
        self.cell_write_energy_scale *= other_vars.get("cell_write_energy_mult", 1)
        # Make sure cols_muxed is set before cols so that you don't get part of the name
        # overwritten
        my_set = [
            "sequential",
            "cols_muxed",
            "rows",
            "cols",
            "technology",
            "num_output_levels",
            "read_pulse_width",
            "voltage",
            "threshold_voltage",
        ]
        for to_set in my_set:
            logger.debug("Setting %s to %s", to_set, getattr(self, to_set))
            cfg = replace_cfg(to_set, getattr(self, to_set), cfg, cfgfile)
        for to_set in [a for a in other_args if a[0] not in my_set]:
            logger.debug("Setting %s to %s", to_set[0], to_set[1])
            if "global_cycle_seconds" not in to_set[0]:
                cfg = replace_cfg(to_set[0], to_set[1], cfg, cfgfile)

        # Write config
        inputpath = os.path.realpath(CFG_WRITE_PATH)
        with open(inputpath, "w") as f:
            f.write(cfg)
        os.chmod(inputpath, 0o777)

        # Run
        logger.info("Running %s %s", NEUROSIM_PATH, inputpath)
        proc = subprocess.Popen(
            [NEUROSIM_PATH, inputpath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        def read_pipe_thread(pipe, write_to: list):
            while proc.poll() is None:
                write_to.append(pipe.read().decode("utf-8"))
            write_to.append(pipe.read().decode("utf-8"))

        stdout, stderr = [], []
        stdout_thread = threading.Thread(
            target=read_pipe_thread, args=(proc.stdout, stdout)
        )
        stderr_thread = threading.Thread(
            target=read_pipe_thread, args=(proc.stderr, stderr)
        )
        stdout_thread.start()
        stderr_thread.start()
        stdout_thread.join()
        stderr_thread.join()
        stdout, stderr = "".join(stdout), "".join(stderr)
        if proc.returncode != 0:
            logger.error("NeuroSIM returned error code %s", proc.returncode)
            logger.error(stderr)
            raise ValueError("NeuroSIM returned error code %s", proc.returncode)
        results = stdout
        logger.debug("NeuroSIM output:\n" + results)
        self.comps = [
            Component(line) for line in results.split("\n") if "<COMPONENT>" in line
        ]
        for c in self.comps:
            if "adc" in c.name:
                c.area *= self.adc_area_scale
                for a in dir(c):
                    if "energy" in a:
                        setattr(c, a, getattr(c, a) * self.adc_energy_scale)

        if not self.comps:
            logger.error("NeuroSIM returned no components. NeuroSIM output below.")
            logger.error("\n\nNeuroSIM returned no components. NeuroSIM output below.")
            logger.error("| " + results.replace("\n", "\n| ") + "  ")
            if err:
                logger.error("| " + err.decode("utf-8").replace("\n", "\n| ") + "  ")
            logger.error("NeuroSIM returned no components. NeuroSIM output above.")
            raise ValueError(
                "NeuroSIM returned no components. Check the generated Neurosim input"
                " config and make sure all values are populated."
            )

        columns_at_once = nvsimget("Columns read at once:", results, False)
        min_latency = nvsimget("Minimum latency per read:", results, False)
        logger.info(
            "Crossbar minimum latency is %s ns to read %s columns at once.",
            min_latency,
            columns_at_once,
        )
        if min_latency < self.cycle_seconds * 1e9:
            logger.warning(
                "Minimum crossbar latency of %s ns is less than the cycle "
                "time of %s ns.",
                min_latency,
                self.cycle_seconds * 1e9,
            )
        # Remove the config file
        os.remove(inputpath)

    def get_components(self, read: bool, hi: bool) -> List[Component]:
        """Returns a list of components matching the criteria"""
        comps = [c for c in self.comps if c.read == read]
        if self.has_adc:
            comps = [c for c in comps if "memcell" not in c.name]
        else:
            comps = [c for c in comps if "adc" not in c.name]
        if hi:
            comps = [c for c in comps if "celllo" not in c.name]
        else:
            comps = [c for c in comps if "cellhi" not in c.name]
        return comps

    def energy_off(self, kind: str, read: bool, hi: bool) -> float:
        """Returns energy of running all peripherals and sending a logic '0'."""
        comps = [c for c in self.get_components(read, hi) if kind in c.name]
        return sum(c.activation_energy for c in comps)

    def energy_on(self, kind: str, read: bool, hi: bool) -> float:
        """Returns the energy of running all peripherals and sending a logic '1'."""
        # Grab all components and filter by matching action
        comps = self.get_components(read, hi)
        on_energy = sum(getattr(c, f"energy_per_{kind.lower()}", 0) for c in comps)
        # Also add in base activation energy of matching components
        return on_energy + self.energy_off(kind, read, hi)

    def energy_cell(self, read: bool, hi: bool) -> float:
        """
        Returns the energy of reading/writing a memory cell depending on HI or LO conductance.
        """
        comps = self.get_components(read, hi)
        return sum(c.energy_per_cell for c in comps)

    def area_per_cell(self) -> float:
        """Returns the area of a single cell."""
        # Cell area is reported several places. Here we just grab it in the write section for a HI
        # cell.
        return sum(c.area for c in self.comps if c.read and "memcell cellhi" in c.name)

    def leakage_peripheral(self) -> float:
        """Returns the leakage of all peripherals."""
        # Grab from the read section because read section has all components. Only 'row' or 'col'
        # components to exlude array (which has everything plus extras we don't want) and cells
        return (
            sum(
                c.leakage
                for c in self.comps
                if "row" in c.name or "col" in c.name and c.read
            )
            * self.global_cycle_seconds
        )

    def leakage_per_cell(self) -> float:
        """Returns the leakage of a single cell."""
        # Cell leakage is reported several places. Here we just grab it in the write section for a
        # HI cell.
        return (
            sum(c.leakage for c in self.comps if c.read and "memcell cellhi" in c.name)
            * self.global_cycle_seconds
        )

    def activation_energy(self, target: str) -> float:
        """Returns the energy of a given misc component."""
        # Report for reads because Neurosim C++ wrapper places misc energy in read section.
        return sum(
            c.activation_energy
            for c in self.comps
            if target.lower() == c.name and c.read
        )

    def area(self, target: str) -> float:
        """Returns the area of a given misc component."""
        comps = self.get_components(True, True)
        return sum(c.area for c in comps if target.lower() in c.name)

    def leakage(self, target: str) -> float:
        """Returns the leakage of a given misc component."""
        comps = self.get_components(True, True)
        return (
            sum(c.leakage for c in comps if target.lower() in c.name)
            * self.global_cycle_seconds
        )


def stats2dict(
    read_energy: float, write_energy: float, area: float, leakage: float
) -> Dict[str, float]:
    """Returns a dictionary of stats."""
    return {
        "Read Energy": read_energy,
        "Write Energy": write_energy,
        "Area": area,
        "Leakage": leakage,
    }


def rowcol_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float, kind: str
) -> Dict[str, float]:
    """Calculates the stats for row or column activations."""
    read_on_cello = crossbar.energy_on(kind, True, False)
    read_on_cellhi = crossbar.energy_on(kind, True, True)
    write_on_cello = crossbar.energy_on(kind, False, False)
    write_on_cellhi = crossbar.energy_on(kind, False, True)
    read_on = read_on_cello + (read_on_cellhi - read_on_cello) * avg_cell
    write_on = write_on_cello + (write_on_cellhi - write_on_cello) * avg_cell

    read_off_cello = crossbar.energy_off(kind, True, False)
    read_off_cellhi = crossbar.energy_off(kind, True, True)
    read_off = read_off_cello + (read_off_cellhi - read_off_cello) * avg_cell

    area = crossbar.area(kind)
    leakage = crossbar.leakage(kind)
    read = read_off + (read_on - read_off) * avg_input
    write = write_on  # Can't gate writes becasuse we still have to reset cells
    return stats2dict(read, write, area, leakage * crossbar.global_cycle_seconds)


def row_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for row energy, area, and leakage."""
    # For temporal DAC non-PWM mode, inputs are activated multiple times. For PWM mode, inputs are
    # held high for longer so no extra switching is needed.
    if crossbar.temporal_spiking:
        avg_input *= crossbar.max_activation_time
    # For voltage DAC, some inputs are activated with a lower voltage
    stats = rowcol_stats(crossbar, avg_input, avg_cell, "row")
    stats_dac = rowcol_stats(crossbar, avg_input, avg_cell, "rowdac")

    # If there are multiple voltage DAC levels, add one row driver for each level.
    # The drivers can be attached to different voltage rails.
    dac_scale = 2 ** (crossbar.voltage_dac_bits - 1) - 1
    stats["Area"] = stats["Area"] + stats_dac["Area"] * dac_scale
    stats["Leakage"] = stats["Leakage"] + stats_dac["Leakage"] * dac_scale
    return stats


def col_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for column energy, area, and leakage."""
    # Don't use average input. Columns are always activated regardless of row input value.
    return rowcol_stats(crossbar, 1, avg_cell, "col")


def cell_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    # Cell energy from driving current through cell
    cell_on = crossbar.energy_cell(True, True)
    cell_off = crossbar.energy_cell(True, False)

    read_memcell_energy = (cell_off + (cell_on - cell_off) * avg_cell) * avg_input
    # Cell energy from driving current through cell
    cell_on = crossbar.energy_cell(False, True)
    cell_off = crossbar.energy_cell(False, False)
    write_memcell_energy = cell_on + (cell_off - cell_on) * avg_cell

    # Cells will have a substantial leakage impact
    # Also multiply read energy by temporal DAC bits
    rlscale = crossbar.cell_read_leak_energy_scale
    act_time = crossbar.max_activation_time
    wscale = crossbar.cell_write_energy_scale
    return stats2dict(
        read_memcell_energy * act_time,
        write_memcell_energy * wscale,
        crossbar.area_per_cell(),
        crossbar.leakage_per_cell() * rlscale,
    )


def misc_stats(crossbar: Crossbar, target: str) -> Dict[str, float]:
    """Returns dictionary of stats for misc component energy, area, and leakage."""
    return stats2dict(
        crossbar.energy_on(target, True, False),
        0,
        crossbar.area(target),
        crossbar.leakage(target),
    )


def adder_tree_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for adder tree energy, area, and leakage."""
    return misc_stats(crossbar, "adder tree")


def adder_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for adder energy, area, and leakage."""
    # Our name-grabbing scheme picks up
    adder_stats = misc_stats(crossbar, "adder")
    adder_tree_stats = misc_stats(crossbar, "adder tree")
    return {k: adder_stats[k] - adder_tree_stats[k] for k in adder_stats}


def shift_add_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for shift add energy, area, and leakage."""
    return misc_stats(crossbar, "shift add")


def max_pool_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for max pool energy, area, and leakage."""
    return misc_stats(crossbar, "maxpool")


def mux_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for mux energy, area, and leakage."""
    return misc_stats(crossbar, "peripheral mux")


def flip_flop_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for flip flop energy, area, and leakage."""
    return misc_stats(crossbar, "flip flop")


def not_gate_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for not gate energy, area, and leakage."""
    return misc_stats(crossbar, "not gate")


def nor_gate_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for nor gate energy, area, and leakage."""
    return misc_stats(crossbar, "nor gate")


def nand_gate_stats(
    crossbar: Crossbar, avg_input: float, avg_cell: float
) -> Dict[str, float]:
    """Returns dictionary of stats for nand gate energy, area, and leakage."""
    return misc_stats(crossbar, "nand gate")

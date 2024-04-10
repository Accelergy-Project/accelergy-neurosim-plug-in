from setuptools import setup
from setuptools.command.build_ext import build_ext
import sys
import subprocess
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CELLS = [
    os.path.join("cells", c) for c in os.listdir(os.path.join(SCRIPT_DIR, "cells"))
]


def readme():
    """Returns readme contents"""
    with open("README.md") as f:
        return f.read()


class Build(build_ext):
    """Calls makefile"""

    def run(self):
        print("Building NeuroSim")
        if subprocess.call(["make", "make"]) != 0:
            sys.exit(-1)
        build_ext.run(self)


setup(
    name="accelergy-neurosim-plug-in",
    version="0.1",
    description="An Accelergy framework plugin for NeuroSim",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    keywords="hardware energy estimation analog adc neurosim pim processing-in-memory cim",
    author="Tanner Andrulis",
    author_email="andrulis@mit.edu",
    license="MIT",
    install_requires=[],
    python_requires=">=3.8",
    data_files=[
        (
            "share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin/",
            ["./accelergywrapper.py", "./neurointerface.py", "./default_config.cfg"],
        ),
        (
            "share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin/",
            ["./neurosim.estimator.yaml"],
        ),
        (
            "share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin/NeuroSim/",
            ["./NeuroSim/main"],
        ),
        ("share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin/cells/", CELLS),
        (
            "share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin/",
            ["accelergywrapper.py", "neurointerface.py", "default_config.cfg"],
        ),
    ],
    py_modules=[],
    entry_points={},
    zip_safe=False,
    cmdclass={
        "build_ext": Build,
    },
)

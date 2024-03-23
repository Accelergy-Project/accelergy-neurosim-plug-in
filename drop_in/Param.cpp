/*******************************************************************************
* Neurosim + MIT-Accelergy project plugin. This work was altered by Tanner Andrulis, 
* and retains the original copyright below.
* Email: Andrulis@mit.edu
* Changes madein this file:
*	Added support for a config file intializing some of the parameters.
*
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "math.h"
#include "Param.h"
#include "constant.h"
#include <regex>

using namespace std;

std::string parseheader(std::string header, std::string content) {
	std::smatch sm;
	if(std::regex_search(content, sm, std::regex(header + ":\\s*([a-zA-Z0-9\\.\\-]+)"))) {
		return sm[1];
	} else {
		cout << "Error: \"" << header << ": <value> \" is not found in the config file." << endl;
		exit(1);
	}
}

double Param::rfloat(std::string header) {
	auto v = parseheader(header, cfgText);
	try {return std::stod(v);}
	catch(...) {}
	cout << "Error: " << header << " is not a valid double: " << v << endl;
	exit(1);
	return 0;
}

int Param::rint(std::string header) {
	auto v = parseheader(header, cfgText);
	try {return std::stoi(v);}
	catch(...) {}
	cout << "Error: " << header << " is not a valid int: " << v << endl;
	exit(1);
	return 0;
}

Param::Param() {}

void Param::Initialize() {
	/***************************************** user defined design options and parameters *****************************************/
	numColPerSynapse = 1; // Unused by Timeloop. Must set or we get a divide by 0 exception.

	// TIMELOOP: Doesn't matter. We just do parallel.
	operationmode = rint("sequential"); // 1: conventionalSequential (Use several multi-bit RRAM as one synapse)
								        // 2: conventionalParallel (Use several multi-bit RRAM as one synapse)
	
	memcelltype = rint("memcelltype"); // 1: cell.memCellType = Type::SRAM
								       // 2: cell.memCellType = Type::RRAM
								       // 3: cell.memCellType = Type::FeFET
	
	accesstype = rint("accesstype"); // 1: cell.accessType = CMOS_access
								     // 2: cell.accessType = BJT_access
								     // 3: cell.accessType = diode_access
								     // 4: cell.accessType = none_access (Crossbar Array)
	
	// TIMELOOP: Doesn't matter. We keep LP conventiaonal transistors.
	transistortype = 1;     	// 1: inputParameter.transistorType = conventional
	deviceroadmap = rint("deviceRoadmap"); // 1: inputParameter.deviceRoadmap = HP
								           // 2: inputParameter.deviceRoadmap = LSTP
								           // -1: LP
	
	// TIMELOOP: Doesn't matter. We don't use NeuroSim buffers.
	globalBufferType = false;    // false: register file
								// true: SRAM
	globalBufferCoreSizeRow = 128;
	globalBufferCoreSizeCol = 128;
	tileBufferType = false;      // false: register file
								// true: SRAM
	tileBufferCoreSizeRow = 32;
	tileBufferCoreSizeCol = 32;
	peBufferType = false;        // false: register file
								// true: SRAM
	chipActivation = true;      // false: activation (reLu/sigmoid) inside Tile
								// true: activation outside Tile
	
	// TIMELOOP: Doesn't matter. This doesn't change LUT parameters.
	reLu = false;                // false: sigmoid
								// true: reLu
	// TIMELOOP: Doesn't matter.
	novelMapping = true;        // false: conventional mapping
								// true: novel mapping
								
	SARADC = rint("SARADC");              // false: MLSA
	                            // true: sar ADC
	currentMode = rint("currentMode");         // false: MLSA use VSA
	                            // true: MLSA use CSA
	
	// Timeloop: Ignore all of these global system characteristics. We'll keep validated at true though.
	pipeline = false;            // false: layer-by-layer process --> huge leakage energy in HP
								// true: pipeline process
	speedUpDegree = 8;          // 1 = no speed up --> original speed
								// 2 and more : speed up ratio, the higher, the faster
								// A speed-up degree upper bound: when there is no leak period during each layer --> no need to further fold the system clock
								// This leak period is defined by IFM sizes and data flow, the actual process latency of each layer may be different due to extra peripheries
	
	validated = true;			// false: no calibration factors
								// true: validated by silicon data (wiring area in layout, gate switching activity, post-layout performance drop...)
								
	synchronous = false;		// false: asynchronous
								// true: synchronous, clkFreq will be decided by sensing delay

	// TIMELOOP: Ignore these algorithm-level characteritics. Handle at tihe Timeloop level
	numBitInput = 1;			// precision of input neural activation
	synapseBit = 1;				// precision of synapse weight
	/*** algorithm weight range, the default wrapper (based on WAGE) has fixed weight range of (-1, 1) ***/
	algoWeightMax = 1;
	algoWeightMin = -1;
	
	// Timeloop: Ignore clock frequency (doesn't matter) and temp (keep at default).
	/*** conventional hardware design options ***/
	clkFreq = 1e9;                      // Clock frequency
	temp = 300;                         // Temperature (K)
	// technode: 130	 --> wireWidth: 175
	// technode: 90		 --> wireWidth: 110
	// technode: 65      --> wireWidth: 105
	// technode: 45      --> wireWidth: 80
	// technode: 32      --> wireWidth: 56
	// technode: 22      --> wireWidth: 40
	// technode: 14      --> wireWidth: 25
	// technode: 10, 7   --> wireWidth: 18
	technode = rint("technode");        // Technology
	
	// Timeloop: Ignore wire width. It is set in main
	wireWidth = 40;                     // wireWidth of the cell for Accuracy calculation
	// Timeloop: Ignore global bus
	globalBusDelayTolerance = 0.1;      // to relax bus delay for global H-Tree (chip level: communication among tiles), if tolerance is 0.1, the latency will be relax to (1+0.1)*optimalLatency (trade-off with energy)
	localBusDelayTolerance = 0.1;       // to relax bus delay for global H-Tree (tile level: communication among PEs), if tolerance is 0.1, the latency will be relax to (1+0.1)*optimalLatency (trade-off with energy)
	treeFoldedRatio = 4;                // the H-Tree is assumed to be able to folding in layout (save area)
	maxGlobalBusWidth = 2048;           // the max buswidth allowed on chip level (just a upper_bound, the actual bus width is defined according to the auto floorplan)
										// NOTE: Carefully choose this number!!!
										// e.g. when use pipeline with high speedUpDegree, i.e. high throughput, need to increase the global bus width (interface of global buffer) --> guarantee global buffer speed

	numRowSubArray = rint("numRowSubArray");               // # of rows in single subArray
	numColSubArray = rint("numColSubArray");               // # of columns in single subArray
	
	// Timeloop: Ignore these. We'll keep them at default.
	/*** option to relax subArray layout ***/
	relaxArrayCellHeight = 0;           // relax ArrayCellHeight or not
	relaxArrayCellWidth = 0;            // relax ArrayCellWidth or not
	
	numColMuxed = rint("numColMuxed");                    // How many columns share 1 ADC (for eNVM and FeFET) or parallel SRAM
	levelOutput = rint("levelOutput");                   // # of levels of the multilevelSenseAmp output, should be in 2^N forms; e.g. 32 levels --> 5-bit ADC
	//Timeloop: Ignore.
	cellBit = 1;                        // precision of memory device 
	
	/*** parameters for SRAM ***/
	// due the scaling, suggested SRAM cell size above 22nm: 160F^2
	// SRAM cell size at 14nm: 300F^2
	// SRAM cell size at 10nm: 400F^2
	// SRAM cell size at 7nm: 600F^2
	featuresize = rfloat("memcellFeatureSize");                // Wire width for subArray simulation
	heightInFeatureSizeSRAM = rfloat("heightInFeatureSizeSRAM");        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = rfloat("widthInFeatureSizeSRAM");        // SRAM Cell width in feature size  
	widthSRAMCellNMOS = rfloat("widthSRAMCellNMOS");                            
	widthSRAMCellPMOS = rfloat("widthSRAMCellPMOS");
	widthAccessCMOS = rfloat("widthAccessCMOS");
	minSenseVoltage = rfloat("minSenseVoltage");
	
	/*** parameters for analog synaptic devices ***/
	heightInFeatureSize1T1R = rfloat("heightInFeatureSize1T1R");        // 1T1R Cell height in feature size
	widthInFeatureSize1T1R = rfloat("widthInFeatureSize1T1R");         // 1T1R Cell width in feature size
	heightInFeatureSizeCrossbar = rfloat("heightInFeatureSizeCrossbar");    // Crossbar Cell height in feature size
	widthInFeatureSizeCrossbar = rfloat("widthInFeatureSizeCrossbar");     // Crossbar Cell width in feature size
	
	resistanceOn = rfloat("resistanceOn");               							// Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	resistanceOff = rfloat("resistanceOff");          					// Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	maxConductance = (double) 1/resistanceOn;
	minConductance = (double) 1/resistanceOff;
	
	readVoltage = rfloat("readVoltage");	                				// On-chip read voltage for memory cell
	readPulseWidth = rfloat("readPulseWidth");             					// read pulse width in sec
	accessVoltage = rfloat("accessVoltage");                					// Gate voltage for the transistor in 1T1R
	resistanceAccess = rfloat("accessTransistorResistance"); 
	writeVoltage = rfloat("writeVoltage");											// Enable level shifer if writeVoltage > 1.5V
	writePulseWidth = rfloat("writePulseWidth"); //2ns
	numWritePulse = rfloat("numWritePulse");

	cellCapacitanceAdjust = rfloat("cellCapacitanceAdjust"); // fF
	cellCapacitanceMultiplier = rfloat("cellCapacitanceMultiplier");
	vdd = rfloat("vdd");
	vth = rfloat("vth");

	/*** Calibration parameters ***/
	if(validated){
		alpha = 1.44;	// wiring area of level shifter
		beta = 1.4;  	// latency factor of sensing cycle
		gamma = 0.5; 	// switching activity of DFF in shifter-add and accumulator
		delta = 0.15; 	// switching activity of adder 
		epsilon = 0.05; // switching activity of control circuits
		zeta = 1.22; 	// post-layout energy increase
	}		
	
	/***************************************** user defined design options and parameters *****************************************/
	
	
	
	/***************************************** Initialization of parameters NO need to modify *****************************************/
	
	if (memcelltype == 1) {
		cellBit = 1;             // force cellBit = 1 for all SRAM cases
	} 
	
	/*** initialize operationMode as default ***/
	conventionalParallel = 0;
	conventionalSequential = 0;
	BNNparallelMode = 0;                
	BNNsequentialMode = 0;              
	XNORsequentialMode = 0;          
	XNORparallelMode = 0;         
	switch(operationmode) {
		case 6:	    XNORparallelMode = 1;               break;     
		case 5:	    XNORsequentialMode = 1;             break;     
		case 4:	    BNNparallelMode = 1;                break;     
		case 3:	    BNNsequentialMode = 1;              break;     
		case 2:	    conventionalParallel = 1;           break;     
		case 1:	    conventionalSequential = 1;         break;     
		default:	printf("operationmode ERROR\n");	exit(-1);
	}
	
	/*** parallel read ***/
	parallelRead = 0;
	if(conventionalParallel || BNNparallelMode || XNORparallelMode) {
		parallelRead = 1;
	} else {
		parallelRead = 0;
	}
	
	/*** Initialize interconnect wires ***/
	switch(wireWidth) {
		case 175: 	AR = 1.60; Rho = 2.20e-8; break;  // for technode: 130
		case 110: 	AR = 1.60; Rho = 2.52e-8; break;  // for technode: 90
		case 105:	AR = 1.70; Rho = 2.68e-8; break;  // for technode: 65
		case 80:	AR = 1.70; Rho = 3.31e-8; break;  // for technode: 45
		case 56:	AR = 1.80; Rho = 3.70e-8; break;  // for technode: 32
		case 40:	AR = 1.90; Rho = 4.03e-8; break;  // for technode: 22
		case 25:	AR = 2.00; Rho = 5.08e-8; break;  // for technode: 14
		case 18:	AR = 2.00; Rho = 6.35e-8; break;  // for technode: 7, 10
		case -1:	break;	// Ignore wire resistance or user define
		default:	exit(-1); puts("Wire width out of range"); 
	}
	
	if (memcelltype == 1) {
		wireLengthRow = wireWidth * 1e-9 * heightInFeatureSizeSRAM;
		wireLengthCol = wireWidth * 1e-9 * widthInFeatureSizeSRAM;
	} else {
		if (accesstype == 1) {
			wireLengthRow = wireWidth * 1e-9 * heightInFeatureSize1T1R;
			wireLengthCol = wireWidth * 1e-9 * widthInFeatureSize1T1R;
		} else {
			wireLengthRow = wireWidth * 1e-9 * heightInFeatureSizeCrossbar;
			wireLengthCol = wireWidth * 1e-9 * widthInFeatureSizeCrossbar;
		}
	}
	Rho *= (1+0.00451*abs(temp-300));
	if (wireWidth == -1) {
		unitLengthWireResistance = 1.0;	// Use a small number to prevent numerical error for NeuroSim
		wireResistanceRow = 0;
		wireResistanceCol = 0;
	} else {
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
		wireResistanceRow = unitLengthWireResistance * wireLengthRow;
		wireResistanceCol = unitLengthWireResistance * wireLengthCol;
	}
	/***************************************** Initialization of parameters NO need to modify *****************************************/
}


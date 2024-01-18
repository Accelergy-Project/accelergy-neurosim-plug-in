/*******************************************************************************
* Neurosim + MIT-Accelergy project plugin. This work was altered by Tanner Andrulis, 
* and retains the original copyright below.
* Email: Andrulis@mit.edu
* Changes madein this file:
*	This is a new top-level file. It initializes components using an input file and calculates
*   energy scaling based on number of rows & columns + number of ON cells.
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

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include "Bus.h"
#include "SubArray.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "DFF.h"
#include "Definition.h"
#include "SubArray.h"
#include "AdderTree.h"
#include "ShiftAdd.h"
#include "MaxPooling.h"
#include "Sigmoid.h"
#include <list>
#include <tuple>

extern Param *param;
SubArray *subArray;
bool genCrossbar;
// compStat entry meanings:
// 	Name, 
// 	element, 
// 	value for storing dynamic energy, 
// 	pointer to read dynamic energy (or NULL for readDynamicEnergy), 
// 	pointer to write dynamic energy, 
// 	#actions (Activation energy scaled down by this number), 
// 	#actions/ON (Activation energy and ON energy both scaled down by this nuber)
// 	If pointers are set, area and latency are zeroed.
typedef std::tuple<std::string, FunctionUnit*, double, double*, double*, double, double> compStat;
typedef std::tuple<double, double, double, double> scaleResults;

void Initalize(int _numRow, int _numCol, InputParameter& inputParameter, Technology& tech, MemCell& cell) {
	// =============================================================================================
	// These top two are from chip.cpp, the previous top level.
	// =============================================================================================
	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}
	// Num row per synapse is unused. Num col per synapse only goes into SubArray's shift+adds, so unused for our purposes
	// UNUSED if (param->XNORparallelMode || param->XNORsequentialMode) {
	// UNUSED 	param->numRowPerSynapse = 2;
	// UNUSED } else {
	// UNUSED 	param->numRowPerSynapse = 1;
	// UNUSED }
	// UNUSED if (param->BNNparallelMode) {
	// UNUSED 	param->numColPerSynapse = 2;
	// UNUSED } else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
	// UNUSED 	param->numColPerSynapse = 1;
	// UNUSED } else {
	// UNUSED 	param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit); 
	// UNUSED }

	// =============================================================================================
    // This is the pieces of the init function from ProcessingUnit.cpp that effect SubArray.cpp
    // Eliminated all extra-subarray pieces like buffers and adder trees
	// =============================================================================================

	/*** circuit level parameters ***/
	switch(param->memcelltype) {
		case 3:     cell.memCellType = Type::FeFET; break;
		case 2:	    cell.memCellType = Type::RRAM; break;
		case 1:	    cell.memCellType = Type::SRAM; break;
		case -1:	break;
		default:	exit(-1);
	}
	switch(param->accesstype) {
		case 4:	    cell.accessType = none_access;  break;
		case 3:	    cell.accessType = diode_access; break;
		case 2:	    cell.accessType = BJT_access;   break;
		case 1:	    cell.accessType = CMOS_access;  break;
		case -1:	break;
		default:	exit(-1);
	}				
					
	switch(param->transistortype) {
		case 3:	    inputParameter.transistorType = TFET;          break;
		case 2:	    inputParameter.transistorType = FET_2D;        break;
		case 1:	    inputParameter.transistorType = conventional;  break;
		case -1:	inputParameter.transistorType = conventional;  break;
		default:	exit(-1);
	}
	
	switch(param->deviceroadmap) {
		case 2:	    inputParameter.deviceRoadmap = LSTP;  break;
		case 1:	    inputParameter.deviceRoadmap = HP;    break;
		case -1:	inputParameter.deviceRoadmap = LP;    break;
		default:	exit(-1);
	}
	

	subArray = new SubArray(inputParameter, tech, cell);
	
	subArray->FPGA = false; // The FPGA set is from Tanner... by default FPGA is true


	/* Create SubArray object and link the required global objects (not initialization) */
	inputParameter.temperature = param->temp;   // Temperature (K)
	inputParameter.processNode = param->technode;    // Technology node

	tech.vdd_override = param->vdd;	// Override default Vdd
	tech.vth_override = param->vth;	// Override default Vth
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);
	param->readVoltage = param->readVoltage == 0 ? tech.vdd : param->readVoltage;
	param->writeVoltage = param->writeVoltage == 0 ? tech.vdd : param->writeVoltage;
	param->accessVoltage = param->accessVoltage == 0 ? tech.vdd : param->accessVoltage;

	cell.resistanceOn = param->resistanceOn;	                                // Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	cell.resistanceOff = param->resistanceOff;	                                // Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;            // Average resistance (for energy estimation)
	cell.readVoltage = param->readVoltage;	                                    // On-chip read voltage for memory cell
	cell.readPulseWidth = param->readPulseWidth;
	cell.accessVoltage = param->accessVoltage;                                       // Gate voltage for the transistor in 1T1R
	cell.resistanceAccess = param->resistanceAccess;
	cell.featureSize = param->featuresize; 
	cell.writeVoltage = param->writeVoltage;
	cell.writePulseWidth = param->writePulseWidth;
	

	if (cell.memCellType == Type::SRAM) {   // SRAM
		printf("%e\n", param->heightInFeatureSizeSRAM);
		printf("%e\n", param->widthInFeatureSizeSRAM);

		cell.heightInFeatureSize = param->heightInFeatureSizeSRAM;                   // Cell height in feature size
		cell.widthInFeatureSize = param->widthInFeatureSizeSRAM;                     // Cell width in feature size
		cell.widthSRAMCellNMOS = param->widthSRAMCellNMOS;
		cell.widthSRAMCellPMOS = param->widthSRAMCellPMOS;
		cell.widthAccessCMOS = param->widthAccessCMOS;
		cell.minSenseVoltage = param->minSenseVoltage;
	} else {
		cell.heightInFeatureSize = (cell.accessType==CMOS_access)? param->heightInFeatureSize1T1R : param->heightInFeatureSizeCrossbar;         // Cell height in feature size
		cell.widthInFeatureSize = (cell.accessType==CMOS_access)? param->widthInFeatureSize1T1R : param->widthInFeatureSizeCrossbar;            // Cell width in feature size
	} 
	cell.widthAccessCMOS = param->widthAccessCMOS;
	subArray->XNORparallelMode = param->XNORparallelMode;               
	subArray->XNORsequentialMode = param->XNORsequentialMode;             
	subArray->BNNparallelMode = param->BNNparallelMode;                
	subArray->BNNsequentialMode = param->BNNsequentialMode;              
	subArray->conventionalParallel = param->conventionalParallel;                  
	subArray->conventionalSequential = param->conventionalSequential;                 
	subArray->numRow = param->numRowSubArray;
	subArray->numCol = param->numRowSubArray;
	subArray->levelOutput = param->levelOutput;
	subArray->numColMuxed = param->numColMuxed;               // How many columns share 1 read circuit (for neuro mode with analog RRAM) or 1 S/A (for memory mode or neuro mode with digital RRAM)
    subArray->clkFreq = param->clkFreq;                       // Clock frequency
	subArray->relaxArrayCellHeight = param->relaxArrayCellHeight;
	subArray->relaxArrayCellWidth = param->relaxArrayCellWidth;
	subArray->numReadPulse = param->numBitInput;
	subArray->avgWeightBit = param->cellBit;
	subArray->numCellPerSynapse = param->numColPerSynapse;
	subArray->SARADC = param->SARADC;
	subArray->currentMode = param->currentMode;
	subArray->validated = param->validated;
	subArray->spikingMode = NONSPIKING;
	

	int numRow = param->numRowSubArray;
	int numCol = param->numColSubArray;
	
	if (subArray->numColMuxed > numCol) {                      // Set the upperbound of numColMuxed
		subArray->numColMuxed = numCol;
	}

	subArray->numReadCellPerOperationFPGA = numCol;	           // Not relevant for IMEC
	subArray->numWriteCellPerOperationFPGA = numCol;	       // Not relevant for IMEC
	subArray->numReadCellPerOperationMemory = numCol;          // Define # of SRAM read cells in memory mode because SRAM does not have S/A sharing (not relevant for IMEC)
	subArray->numWriteCellPerOperationMemory = numCol/8;       // # of write cells per operation in SRAM memory or the memory mode of multifunctional memory (not relevant for IMEC)
	subArray->numReadCellPerOperationNeuro = numCol;           // # of SRAM read cells in neuromorphic mode
	subArray->numWriteCellPerOperationNeuro = numCol;	       // For SRAM or analog RRAM in neuro mode
    subArray->maxNumWritePulse = subArray->numWritePulse = param->numWritePulse;

	subArray->Initialize(numRow, numCol, param->unitLengthWireResistance);        // initialize subArray

	// =============================================================================================
	// This one is from ProcessingUnit.cppm, in the calculatePerformance function
	// =============================================================================================
    int cellRange = pow(2, param->cellBit);
    if (param->parallelRead) {
        subArray->levelOutput = param->levelOutput;               // # of levels of the multilevelSenseAmp output
    } else {
        subArray->levelOutput = cellRange;
    }

	subArray->CalculateArea();
}

bool wireAwareProgramming = true;
vector<double> GetColumnResistance(MemCell& cell, int nCellsOn, int nRowsConnected, int nColReads) {
	double columnG = 0; 
	double weight = cell.resistanceOn;
    double row = subArray->numRow;
    double col = subArray->numCol;
    double resCellAccess = subArray->resCellAccess;
    bool parallelRead = param->parallelRead;

	wireAwareProgramming = true;
	// Set wireAwareProgramming based on the environment variable ACCELERGY_NEUROSIM_PLUG_IN_WIRE_AWARE_PROGRAM
	wireAwareProgramming = getenv("ACCELERGY_NEUROSIM_WIRE_UNAWARE") == NULL; 
	
    // CHANGES MADE: Removed loop through all weights.
	// From Neurosim Chip.cpp:
	// 	conductance = cellvalue/(cellrange-1) * (maxConductance-minConductance) + minConductance;

	// If wire aware programming is on:
    // Assumption: The very corner cell has the highest wire resistance. If we program
    //             this cell with MAX conductance, the resistance of this cell + the wire resistance
    //             is the lowest resistance we can program ANY cell (to ensure fidelity of analog
    //             compute)
	// We:
	// 	Calculate max conductance is conductance of farthest corner cell. This cell has the lowest
	//  conductance due to high wire resistance.
	//  Calculate min conductance is conductance of the closest corner cell. This cell has the
	//  highest conductance due to low wire resistance.
	//  Bounded by our worst case cells, we now know that ~all~ cells can be mapped to any
	//  conductance in this range. We spread values linearly from min to max conductance.
    if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {	// eNVM
        double totalWireResistance;
		totalWireResistance = row * param->wireResistanceRow + col * param->wireResistanceCol;
		double accessResistance = cell.accessType == CMOS_access ? cell.resistanceAccess : 0;

		// Wire unaware. Calculate for average wire resistance
		double minConductance = 1.0 / (cell.resistanceOff + totalWireResistance / 2 + accessResistance);
		double maxConductance = 1.0 / (cell.resistanceOn + totalWireResistance / 2 + accessResistance);
		//printf("minConductance = %e\n", minConductance);
		//printf("maxConductance = %e\n", maxConductance);
		// Wire aware. Min conductance of closest cell, max conductance of farthest cell
		if(wireAwareProgramming) {
			minConductance = 1.0 / (totalWireResistance + cell.resistanceOff + accessResistance);
			maxConductance = 1.0 / (totalWireResistance + cell.resistanceOn + accessResistance);
		}
        columnG += minConductance * (subArray->numRow - nCellsOn) + maxConductance * nCellsOn;

		columnG /= subArray->numRow; // Average conductance
		columnG *= nRowsConnected; // Multiplied by number of connected rows
    } else if (cell.memCellType == Type::SRAM) {	
        // SRAM: weight value do not affect sense energy --> read energy calculated in subArray.cpp (based on wireRes wireCap etc)
        double totalWireResistance = (double) (resCellAccess + row * param->wireResistanceCol);
        columnG += (double) 1.0/totalWireResistance;
		columnG *= nRowsConnected;
    }

	// If we're activating rows sequentially, consider the average case per row
	if(param->conventionalSequential && nRowsConnected > 0) columnG /= (double) nRowsConnected;
	
    vector<double> resistance;
	for(int i = 0; i < nColReads; i++) resistance.push_back(1 / columnG);
	return resistance;
}

void CalculateEnergy(MemCell& cell, int nRowReads, int nRowWrites, int nColReads, int nColWrites, int nCellsOn) {
	subArray->activityRowRead = ((double) nRowReads) / subArray->numRow;
    subArray->activityRowWrite = ((double) nRowWrites) / subArray->numRow;
	subArray->activityColRead = ((double) nColReads) / subArray->numCol;
    subArray->activityColWrite = ((double) nColWrites) / subArray->numCol;
    auto columnResistance = GetColumnResistance(cell, nCellsOn, nRowReads, max(nColReads, nColWrites));
	// WARNING: RETURNED POWER IS POWER FOR READING ALL COLUMNS
	subArray->CalculateLatency(columnResistance[0], columnResistance, true);
	subArray->CalculatePower(columnResistance);
}

scaleResults CalculateScaling(compStat c, bool isWrite, int numCellsOn) {
	// Reset the function unit
	auto fu = std::get<1>(c);
	fu->area = fu->emptyArea = fu->usedArea = fu->totalArea = 0;
	fu->readLatency = fu->writeLatency = 0;
	fu->readDynamicEnergy = fu->writeDynamicEnergy = 0;
	fu->leakage = 0;
	subArray->CalculateArea();

	// Set up interesting variables
	int rowreads = 0, colreads = 0, rowwrites = 0, colwrites = 0;
	int* scalerow = isWrite ? &rowwrites : &rowreads;
	int* scalecol = isWrite ? &colwrites : &colreads;
	double* energy = std::get<3>(c) ? std::get<3>(c) : &fu->readDynamicEnergy;
	if(isWrite) energy = std::get<4>(c) ? std::get<4>(c) : &fu->writeDynamicEnergy;
	*energy = 0;
	int nrow = subArray->numRow;
	int ncol = subArray->numCol;
	
	// Energy = Base energy + Energy / Row + Energy / Col + Energy / Cell

	// Calculate scaling based on active rows and cols
	*scalerow = *scalecol = 1;
	CalculateEnergy(cell, rowreads, rowwrites, colreads, colwrites, numCellsOn);
	double base = *energy; // Base + 1 row + 1 col + 1 cell
	*scalerow = nrow;
	CalculateEnergy(cell, rowreads, rowwrites, colreads, colwrites, numCellsOn);
	double row = max((*energy - base) / ((double) nrow - 1), 0.0); // Base + N row + nrows cells -> 1 row + 1 cell
	*scalerow = 1;
	*scalecol = ncol;
	CalculateEnergy(cell, rowreads, rowwrites, colreads, colwrites, numCellsOn);
	double col = max((*energy - base) / ((double) ncol - 1), 0.0); // Base + N col + ncols cells -> 1 col + 1 cell
	*scalerow = nrow;
	*scalecol = ncol;
	CalculateEnergy(cell, rowreads, rowwrites, colreads, colwrites, numCellsOn);
	double prod = max((*energy - row * nrow - col * ncol - base) / ((double) nrow - 1) / ((double) ncol - 1), 0.0); // Base + M row + N col + MN cell -> 1 cell
	row =  max(row - prod, 0.0) / get<6>(c); // 1 row + 1 cell -> 1 row
	col =  max(col - prod, 0.0) / get<6>(c); // 1 col + 1 cell -> 1 col
	base = max(base - row - col - prod, 0.0) / get<5>(c) / get<6>(c); // Base + 1 row + 1 col + 1 cell -> Base

	// Pointers were passed for energy values. If we can't use the default location for energy,
	// we won't use the default location for area/leakage either.
	if(std::get<3>(c) || std::get<4>(c)) {
		fu->area = fu->emptyArea = fu->usedArea = fu->totalArea = 0;
		fu->readLatency = fu->writeLatency = 0;
		fu->leakage = 0;
	}
	return scaleResults(base, row, col, prod);
};

void printHeading() {
	printf("            R/W  , Name                     ,       pJ/act,    pJ/row ON,    ");
	printf("pJ/col ON,  pJ/cross ON,            um^2 area,           pW leakage,         ");
	printf("  ps latency\n");
}

void printStats(std::string name, double eAct, double eRow, double eCol, double eCell, double area, double leakage, double latency, bool isWrite) {
	if (isWrite) printf("<COMPONENT> Write, ");
	else         printf("<COMPONENT> Read , ");
	printf("%-25s, %12e, %12e, %12e, %12e", name.c_str(), eAct*1e12, eRow*1e12, eCol*1e12, eCell*1e12);
	printf(", %20e, %20e, %20e\n", area*1e12, leakage*1e12, latency*1e12);
}
void printStats(std::string name, FunctionUnit fu, scaleResults s, bool isWrite) {
	double latency = isWrite ? fu.writeLatency : fu.readLatency;
	printStats(name, get<0>(s), get<1>(s), get<2>(s), get<3>(s), fu.area, fu.leakage, latency, isWrite);
}

void CalculateEnergy(MemCell& cell) {
	double rampInput = 1e20; // Misc components
	double forceWidth = 0; // Misc components
	double forceHeight = 0; // Misc components
	AreaModify areaModify = NONE; // Misc components
	int numActions = 1; // Misc components
	double capLoad = 0; // Misc components
	int numComponent = 1;
	double clkFreq = 1e9; // I don't think this is used
	
	FunctionUnit* target;

	int precision = param->rint("precision");
	int SA_input_precision = param->rint("shift_register_precision");
	int maxPoolWindow = param->rint("pool_window");
	bool sramsigmoid = param->memcelltype == Type::SRAM; // false for RRAM
	int adder_tree_inputs = param->rint("n_adder_tree_inputs");
	int mux_inputs = param->rint("n_mux_inputs");

	// compStat entry meanings:
	// 	Name, 
	// 	element, 
	// 	value for storing dynamic energy, 
	// 	pointer to read dynamic energy (or NULL for readDynamicEnergy), 
	// 	pointer to write dynamic energy, 
	// 	#actions (Activation energy scaled down by this number), 
	// 	#actions/ON (Activation energy and ON energy both scaled down by this nuber)
	// 	If pointers are set, area and latency are zeroed.
	std::vector<compStat> xbarComps;
	// Name codes:
	//		Array: Array
	//		RowDr: Row components
	//		RowDrDAC: Row components + multiplied for multiple input voltage levels
	//		ColRd: Column components
	xbarComps.push_back(compStat("Array", 		  		subArray, 							0, &subArray->readDynamicEnergyArray, &subArray->writeDynamicEnergyArray, subArray->numCol, subArray->numColMuxed));
	xbarComps.push_back(compStat("Row(WLDec)",   		&subArray->wlDecoder, 				0, NULL, NULL, subArray->numRow, subArray->numColMuxed));
	xbarComps.push_back(compStat("RowDAC(WLDrvNew)",  	&subArray->wlNewDecoderDriver,		0, NULL, NULL, subArray->numRow, subArray->numColMuxed));
	xbarComps.push_back(compStat("RowDAC(WLDrv)",		&subArray->wlDecoderDriver, 		0, NULL, NULL, subArray->numRow, subArray->numColMuxed));
	xbarComps.push_back(compStat("Row(WLSwch)", 		&subArray->wlSwitchMatrix, 			0, NULL, NULL, subArray->numRow, subArray->numColMuxed));
	xbarComps.push_back(compStat("RowDAC(WLSwchNew)",	&subArray->wlNewSwitchMatrix, 		0, NULL, NULL, subArray->numRow, subArray->numColMuxed));
	xbarComps.push_back(compStat("Row(SRAMWR)",			&subArray->sramWriteDriver,			0, NULL, NULL, subArray->numRow, 1));
	xbarComps.push_back(compStat("Col(Mux)",			&subArray->mux, 					0, NULL, NULL, subArray->numCol, 1));
	xbarComps.push_back(compStat("Col(MuxDec)", 		&subArray->muxDecoder,				0, NULL, NULL, subArray->numCol, 1));
	xbarComps.push_back(compStat("Col(SLSwch)", 		&subArray->slSwitchMatrix,			0, NULL, NULL, subArray->numCol, 1));
	std::vector<compStat> adcComps;
	adcComps.push_back(compStat("Col|ADC(SAR)", 		&subArray->sarADC, 					0, NULL, NULL, subArray->numCol, 1));
	adcComps.push_back(compStat("Col|ADC(MLSA)",		&subArray->multilevelSenseAmp, 		0, NULL, NULL, subArray->numCol, 1));
	adcComps.push_back(compStat("Col|ADC(MLSAEnc)",		&subArray->multilevelSAEncoder, 	0, NULL, NULL, subArray->numCol, 1));
	adcComps.push_back(compStat("Col|ADC(SA)",			&subArray->senseAmp, 				0, NULL, NULL, subArray->numCol, 1));


	if(genCrossbar) {
		printf("========================================================================================================================================================================================\n");
		printf("READ: %d ROW, %d COL\n", subArray->numRow, subArray->numCol);
		printf("========================================================================================================================================================================================\n");
		printHeading();
		// This block all has activation multipliled by numColMuxed internally. We remove this to
		// because Accelergy handles it.
		double muxScale = 1.0 / (double) subArray->numColMuxed;

		for(auto& c: xbarComps) {
			auto s = CalculateScaling(c, false, 1);
			FunctionUnit fu = *std::get<1>(c);
			printStats(std::get<0>(c), fu, s, false);
		}
		for(auto& c: adcComps) {
			auto s = CalculateScaling(c, false, subArray->numRow);
			FunctionUnit fu = *std::get<1>(c);
			printStats(std::get<0>(c) + " CELLHI", fu, s, false);
			s = CalculateScaling(c, false, 0);
			fu = *std::get<1>(c);
			printStats(std::get<0>(c) + " CELLLO", fu, s, false);
		}

		printf("WARNING: READ S+A AND SAR ADC ENERGY ALREADY CALCULATE CELL READ ENERGY. USE CUSTOM READ ENERGY IF USER SPECIFIES THEIR OWN A/D CONVERTER.\n");
		// OFF: One connected cell OFF. ON: One connected cell ON.
		// Don't worry about half-selected cells because their bitlines/wordlines can stay floating
		double cellReadEnergyLo = cell.readPulseWidth * pow(cell.readVoltage, 2) / GetColumnResistance(cell, 0, 1, 1)[0];
		double cellReadEnergyHi = cell.readPulseWidth * pow(cell.readVoltage, 2) / GetColumnResistance(cell, subArray->numRow, 1, 1)[0];
		double cellLeakage = 0;
		if(cell.memCellType == Type::SRAM) {
			// Copied from subarray
			cellReadEnergyLo = cellReadEnergyHi = 0;
			cellLeakage = CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
						cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
		}
		double cellArea = subArray->widthArray * subArray->heightArray / subArray->numRow / subArray->numCol;
		printStats("Memcell CELLHI Sel.", 0, 0, 0, cellReadEnergyHi, cellArea, cellLeakage, 0, false);
		printStats("Memcell CELLLO Sel.", 0, 0, 0, cellReadEnergyLo, cellArea, cellLeakage, 0, false);
	}

	if(genCrossbar) {
		printf("========================================================================================================================================================================================\n");
		printf("WRITE: %d ROW, %d COL\n", subArray->numRow, subArray->numCol);
		printf("========================================================================================================================================================================================\n");
		printHeading();

		for(auto& c: xbarComps) {
			auto s = CalculateScaling(c, true, 1);
			FunctionUnit fu = *std::get<1>(c);
			printStats(std::get<0>(c), fu, s, true);
		}

		// OFF: One connected cell OFF. ON: One connected cell ON.
		double cellWriteEnergyLo = cell.writePulseWidth * pow(cell.writeVoltage, 2) * subArray->numWritePulse / GetColumnResistance(cell, 0, 1, 1)[0];
		double cellWriteEnergyHi = cell.writePulseWidth * pow(cell.writeVoltage, 2) * subArray->numWritePulse / GetColumnResistance(cell, subArray->numRow, 1, 1)[0];
		cellWriteEnergyLo += tech.vdd * tech.vdd * subArray->capRow2 * subArray->numWritePulse / subArray->numCol; // Assume all columns are written together
		cellWriteEnergyHi += tech.vdd * tech.vdd * subArray->capRow2 * subArray->numWritePulse / subArray->numCol; // Assume all columns are written together
		double cellLeakage = 0;
		double cellWriteEnergyLoHalfSelected = cellWriteEnergyLo / 4; // Half voltage -> v^2 is quartered
		double cellWriteEnergyHiHalfSelected = cellWriteEnergyHi / 4;
		if(cell.accessType == CMOS_access) cellWriteEnergyLoHalfSelected = cellWriteEnergyHiHalfSelected = 0;
		if(cell.memCellType == Type::SRAM) {
			// Neurosim charges SRAM gate flip energy as array energy. Calculate by flipping exactly
			// One SRAM cell
			cellWriteEnergyLoHalfSelected = cellWriteEnergyHiHalfSelected = 0;
			cellLeakage = CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
						cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			CalculateEnergy(cell, 0, 1, 0, 1, 0);
			cellWriteEnergyLo += subArray->writeDynamicEnergyArray;
			cellWriteEnergyHi += subArray->writeDynamicEnergyArray;
		}
		double cellArea = subArray->widthArray * subArray->heightArray / subArray->numRow / subArray->numCol;
		printStats("Memcell CELLHI Sel.", 0, 0, 0, cellWriteEnergyHi, cellArea, cellLeakage, 0, true);
		printStats("Memcell CELLLO Sel.", 0, 0, 0, cellWriteEnergyLo, cellArea, cellLeakage, 0, true);
		//printStats("Cell ON Half Sel.", 0, 0, 0, cellWriteEnergyHiHalfSelected, cellArea, cellLeakage, 0, true);
		//printStats("Cell OFF Half Sel.", 0, 0, 0, cellWriteEnergyLoHalfSelected, cellArea, cellLeakage, 0, true);
	}
	
	bool printMiscComponents = true;
	if(printMiscComponents) {
		printf("========================================================================================================================================================================================\n");
		printf("MISC COMPONENTS\n");
		printf("========================================================================================================================================================================================\n");

		Adder adder = Adder(inputParameter, tech, cell);
		adder.Initialize(precision, numComponent, clkFreq);
		adder.CalculateArea(forceWidth, forceHeight, areaModify);
		adder.CalculateLatency(rampInput, capLoad, numActions);
		adder.CalculatePower(numActions, numComponent);
		printStats("Adder", adder.readDynamicEnergy, 0, 0, 0, adder.area, adder.leakage, adder.readLatency, false);
			       
				   
		MaxPooling maxPooling = MaxPooling(inputParameter, tech, cell);
		maxPooling.Initialize(precision, maxPoolWindow, numComponent, clkFreq);
		maxPooling.CalculateUnitArea(areaModify);
		maxPooling.CalculateArea(1); // Arg unused, must be nonzero or arithmetic exception
		maxPooling.CalculateLatency(rampInput, capLoad, numActions);
		maxPooling.CalculatePower(numActions);
		printStats("Maxpool", maxPooling.readDynamicEnergy, 0, 0, 0, maxPooling.area, maxPooling.leakage, maxPooling.readLatency, false);

		ShiftAdd shiftAdd = ShiftAdd(inputParameter, tech, cell);
		shiftAdd.Initialize(numComponent, precision, clkFreq, NONSPIKING, SA_input_precision);
		shiftAdd.CalculateArea(forceWidth, forceHeight, areaModify);
		shiftAdd.area = shiftAdd.adder.area + shiftAdd.dff.area; // Shift add calculates dimensions based on other components... 
		                                                         // Area calculation creates weird issues when using S+A alone, so do it ourselves.
		shiftAdd.CalculateLatency(numActions);                   
		shiftAdd.CalculatePower(numActions);
		printStats("Shift Add", shiftAdd.readDynamicEnergy, 0, 0, 0, shiftAdd.area, shiftAdd.leakage, shiftAdd.readLatency, false);

		// Custom adder tree to allow increasing the precision with each level
		int adderTreePrecision = precision;
		int currentInputs = adder_tree_inputs;
		float readDynamicEnergy = 0, area = 0, leakage = 0, readLatency = 0;
		while(currentInputs > 1) {
			currentInputs /= 2; // Now = to number of adders in this level
			adderTreePrecision += 1;
			Adder adder = Adder(inputParameter, tech, cell);
			adder.Initialize(adderTreePrecision, currentInputs, clkFreq);  // Set numComponent to currentInputs to get power of adder
			adder.CalculateArea(forceWidth, forceHeight, areaModify);
			adder.CalculateLatency(rampInput, capLoad, numActions);
			adder.CalculatePower(numActions, currentInputs); // Set numComponent to currentInputs to get power of adder
			readDynamicEnergy += adder.readDynamicEnergy;
			area += adder.area;
			leakage += adder.leakage;
			readLatency += adder.readLatency;
		}
		printStats("Adder Tree", readDynamicEnergy, 0, 0, 0, area, leakage, readLatency, false);

		Mux mux = Mux(inputParameter, tech, cell);
		RowDecoder muxDecoder = RowDecoder(inputParameter, tech, cell);
		mux.Initialize(precision, mux_inputs, 0, true);
		muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(mux_inputs)), true, false);
		mux.CalculateArea(forceWidth, forceHeight, areaModify);
		muxDecoder.CalculateArea(forceWidth, forceHeight, areaModify);
		mux.CalculateLatency(rampInput, capLoad, numActions);
		muxDecoder.CalculateLatency(rampInput, capLoad, capLoad, numActions, 0);
		mux.CalculatePower(numActions);
		muxDecoder.CalculatePower(numActions, 0);
		printStats("Peripheral Mux", mux.readDynamicEnergy + muxDecoder.readDynamicEnergy, 0, 0, 0, mux.area + muxDecoder.area, mux.leakage + muxDecoder.leakage, mux.readLatency + muxDecoder.readLatency, false);

		DFF dff = DFF(inputParameter, tech, cell);
		dff.Initialize(precision, clkFreq);
		dff.CalculateArea(forceWidth, forceHeight, areaModify);
		dff.CalculateLatency(rampInput, numActions);
		dff.CalculatePower(numActions, precision, true);
		printStats("Flip Flop", dff.readDynamicEnergy, 0, 0, 0, dff.area, dff.leakage, dff.readLatency, false);

		// Gate code is taken from MaxPooling.cpp
		double gatew, gateh;
		// INV
		double widthInvN = MIN_NMOS_SIZE * tech.featureSize;
		double widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
		EnlargeSize(&widthInvN, &widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &gateh, &gatew);
		printStats("NOT gate", 0, 0, 0, 0, gatew * gateh, 0, 0, false);

		// NAND
		double widthNandN = 2*MIN_NMOS_SIZE * tech.featureSize;
		double widthNandP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
		EnlargeSize(&widthNandN, &widthNandP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);
		CalculateGateArea(NAND, 2, widthNandN, widthNandP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &gateh, &gatew);
		printStats("NAND gate", 0, 0, 0, 0, gatew * gateh, 0, 0, false);
		
		// NOR1
		double widthNorN = 4*MIN_NMOS_SIZE * tech.featureSize;
		double widthNorP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
		EnlargeSize(&widthNorN, &widthNorP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);
		CalculateGateArea(NOR, 2, widthNorN, widthNorP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &gateh, &gatew);
		printStats("NOR gate", 0, 0, 0, 0, gatew * gateh, 0, 0, false);

	}
	bool printOther = true;
	if(printOther) {
		printf("========================================================================================================================================================================================\n");
		printf("OTHER\n");
		printf("========================================================================================================================================================================================\n");
		printf("Wordline capacitance: %efF\n", max(subArray->capRow1, subArray->capRow2)*1e15);
		printf("Bitline capacitance: %efF\n", subArray->capCol*1e15);
		auto colResistance = GetColumnResistance(cell, subArray->numRow / 2, subArray->numRow, 1);
		subArray->CalculateLatency(rampInput, colResistance, true);
		printf("Columns read at once: %d\n", subArray->numCol / subArray->numColMuxed);
		printf("Minimum latency per read: %.3f ns\n", subArray->readLatency * 1e9);
	}
	printf("========================================================================================================================================================================================\n");
	printf("========================================================================================================================================================================================\n");

}

int main(int argc, char *argv[]) {
	printf("================================================================================\n");
	printf("MODIFIED Neurosim V1.3 for MIT PIM in Accelergy + Timeloop project.\n");
	printf("Changes made to the original DNN Neurosim 1.3 code:\n");
	printf("\tMoved initialization to new main file.\n");
	printf("\tAdded config file reading to the parameter file.\n");
	printf("\tDisabled memristor CMOS access width calculation, allowing users to specify.\n");
	printf("\tCreated top-level file that calculates energy scaling for each component.\n");
	printf("\tCreated Accelergy interfacing scripts.\n");
	printf("\tAdded SRAM cell access energy calculation for SRAM pim designs.\n");
	printf("================================================================================\n");

	// RENAMED MAIN.CPP to main.old to avoid name conflicts
	// Must include definition.h in this file
	// Removed -O3 from CXXFLAGS in makefile for speedy compilation while I'm working on this
	if(argc == 2) genCrossbar = true;
	else if(argc == 3 && !std::string(argv[2]).compare("-m")) genCrossbar = false;
	else {
		cout << "Usage: ./main <config file path> [-m Misc components only, don't generate crossbar.]" << endl;
		exit(-1);
	}

	char* path = argv[1];
	std::ifstream t(path);
	if(!t.good()) {
		cout << "Error: Could not read file  " << path << endl;
		exit(-1);
	}
	cout << "Reading config file " << path << endl;
	std::stringstream buffer;
	buffer << t.rdbuf();
	param->cfgText = buffer.str();
	param->Initialize();

	Initalize(param->numRowSubArray, param->numColSubArray, inputParameter, tech, cell);
	CalculateEnergy(cell);
}

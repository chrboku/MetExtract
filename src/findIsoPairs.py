#    MetExtract II
#    Copyright (C) 2015
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from __future__ import absolute_import, division, print_function
import base64
import logging
import os
import platform
import time
import traceback
from . import HCA_general, Baseline, exportAsFeatureML
from .utils import CallBackMethod, getNormRatio, getDBSuffix
from .Chromatogram import Chromatogram
from .formulaTools import formulaTools, getIsotopeMass, calcIsoEnrichment, isotopologLabel
from .mePyGuis.TracerEdit import ConfiguredTracer
from .MZHCA import HierarchicalClustering, cutTreeSized
from .PolarsDB import PolarsDB
from .findIsoPairs_matchPartners import matchPartners
from .SGR import SGRGenerator
from .chromPeakPicking.peakpickers import BasePeakPicker, filter_peaks
import numpy as np
import polars as pl
import scipy
from pickle import dumps
from copy import copy
from math import floor
from .utils import (
    Bunch,
    ChromPeakPair,
    corr,
    getAtomAdd,
    getDBFormat,
    getSubGraphs,
    mean,
    sd,
    smoothDataSeries,
    weightedMean,
    weightedSd,
)


# returns the abbreviation for a given element
# e.g. element="Carbon" --> return="C"
def getShortName(element):
    fT = formulaTools()
    for i in fT.elemDetails:
        d = fT.elemDetails[i]
        if d[0] == element:
            return d[1]
    return ""


# counts how ofter an entry i is listed in the vector x and returns it as a dictionary
# e.g. x=[1,2,1,3,4,4,5,4] --> return={1:2, 2:1, 3:1, 4:3, 5:1}
def countEntries(x):
    ret = {}
    for i in x:
        if i not in ret:
            ret[i] = 0
        ret[i] = ret[i] + 1
    return ret


def getCorrelationShifted(xic1, xic2, peakStartMin, peakEndMin, rtShiftsMin=None):
    if rtShiftsMin is None:
        raise RuntimeError("parameter rtShiftsMin must be an array of retention time shifts in minutes, e.g., [-3.0, -2.6, -2.2, ... 2.2, 2.6, 3.0]")

    if type(rtShiftsMin) != list and type(rtShiftsMin) != np.ndarray:
        raise RuntimeError("parameter rtShiftsMin must be an array/numpy ndarray of retention time shifts in minutes, e.g., [-3.0, -2.6, -2.2, ... 2.2, 2.6, 3.0]")

    assert xic1.shape[0] == xic2.shape[0]

    corrs = []
    nonShiftCor = correlationFor(xic1, xic2, peakStartMin, peakEndMin)

    for rtShiftMini, rtShiftMin in enumerate(rtShiftsMin):
        ## shift second EIC by current rtShiftMin
        xic2_ = np.copy(xic2)
        xic2_[:, 0] = xic2_[:, 0] + rtShiftMin

        ## get new rts
        newRTs = np.sort(np.concatenate((xic1[:, 0], xic2_[:, 0])))

        xic1_ = interpolateToNewRTs(np.copy(xic1), np.arange(np.min(newRTs), np.max(newRTs) + 0.0001, 0.01))
        xic2_ = interpolateToNewRTs(xic2_, np.arange(np.min(newRTs), np.max(newRTs) + 0.0001, 0.01))
        ## calculate correlation using peak 1
        c = correlationFor(xic1_, xic2_, peakStartMin, peakEndMin)
        corrs.append(c)

    return nonShiftCor, np.max(corrs), rtShiftsMin[np.argmax(corrs)]


def interpolateToNewRTs(xic, newRTs):
    newRTs = np.unique(newRTs)

    nxic = np.zeros((len(newRTs), 2), dtype=xic.dtype)
    for curPosN, rt in enumerate(newRTs):
        nxic[curPosN, 0] = rt

        if rt <= xic[0, 0]:
            nxic[curPosN, 1] = xic[0, 1]

        elif rt >= xic[xic.shape[0] - 1, 0]:
            nxic[curPosN, 1] = xic[xic.shape[0] - 1, 1]
        else:
            bestNextRtInd = np.argmin(np.abs(rt - xic[:, 0]))

            if xic[bestNextRtInd, 0] == rt:
                nxic[curPosN, :] = xic[bestNextRtInd, :]

            else:
                leftInd = None
                rightInd = None

                if xic[bestNextRtInd, 0] < rt:
                    leftInd = bestNextRtInd
                    rightInd = bestNextRtInd + 1

                elif xic[bestNextRtInd, 0] > rt:
                    leftInd = bestNextRtInd - 1
                    rightInd = bestNextRtInd

                assert leftInd is not None, "Branch should not exists..."

                nxic[curPosN, 1] = xic[leftInd, 1] + (xic[rightInd, 1] - xic[leftInd, 1]) / (xic[rightInd, 0] - xic[leftInd, 0]) * (rt - xic[leftInd, 0])

    return np.copy(nxic)


def smooth(abundances):
    algorithm = "rollingaverage"
    if algorithm.lower() == "rollingaverage":
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(abundances, kernel, mode="same")
        return data_convolved

    elif algorithm.lower() == "savgol":
        window_length = self.smoothingwindow_length
        polynom_degree = self.smoothingpolynom_degree
        return scipy.signal.savgol_filter(abundances, window_length, polynom_degree)

    else:
        raise RuntimeError("Unknown algorithm '%s' for smoothing" % (algorithm))


def correlationFor(xic1, xic2, peakStartMin, peakEndMin):
    algorithm = "pearson"

    leftInd = np.argmin(np.abs(xic1[:, 0] - peakStartMin))
    rightInd = np.argmin(np.abs(xic1[:, 0] - peakEndMin))

    if algorithm == "pearson":
        try:
            return corr(xic1[leftInd:rightInd, 1], xic2[leftInd:rightInd, 1])
        except Exception as ex:
            print("There was an error calculating the pearson correlation between the two provided XICs. these are")
            print("XIC 1")
            print(xic1)
            print("")
            print("XIC 2")
            print(xic2)
            print("")
            print("Peak start and end are", peakStartMin, peakEndMin)
            raise RuntimeError("Could not calcualte correlation: '%s'" % (ex))

    else:
        raise RuntimeError("Unknown algorithm '%s' for smoothing" % (algorithm))


peakAbundanceUseSignals = 5
peakAbundanceUseSignalsSides = int((peakAbundanceUseSignals - 1) / 2)


# This class is used as a Command for each LC-HRMS file and is called by the multiprocessing module in MExtract.py
# during the processing of each individual LC-HRMS data files
class FindIsoPairs:
    # Constructor of the class, which stores the processing parameters
    # writeMZXML: 0001: 12C  0010: 12C-Iso  0100: 13C-Iso  1000: 13C
    def __init__(
        self,
        file,
        exOperator="",
        exExperimentID="",
        exComments="",
        exExperimentName="",
        writeFeatureML=False,
        writeTSV=False,
        writeMZXML=9,
        metabolisationExperiment=False,
        labellingisotopeA="12C",
        labellingisotopeB="13C",
        useCIsotopePatternValidation=False,
        xOffset=1.00335,
        minRatio=0,
        maxRatio=9999999,
        useRatio=False,
        configuredTracer=None,
        intensityThreshold=0,
        intensityCutoff=0,
        maxLoading=1,
        xCounts="",
        ppm=2.0,
        isotopicPatternCountLeft=2,
        isotopicPatternCountRight=2,
        lowAbundanceIsotopeCutoff=True,
        intensityThresholdIsotopologs=1000,
        intensityErrorN=0.25,
        intensityErrorL=0.25,
        purityN=0.99,
        purityL=0.99,
        minSpectraCount=1,
        clustPPM=8.0,
        chromPeakPPM=5.0,
        snrTh=1.0,
        scales=[1, 35],
        peakCenterError=5,
        peakScaleError=3,
        minPeakCorr=0.85,
        checkPeaksRatio=False,
        minPeaksRatio=0,
        maxPeaksRatio=99999999,
        checkPeakWidthFilter=False,
        minPeakWidth=0.0,
        maxPeakWidth=9999.0,
        minFWHM=0.0,
        maxFWHM=9999.0,
        calcIsoRatioNative=1,
        calcIsoRatioLabelled=-1,
        calcIsoRatioMoiety=1,
        startTime=2,
        stopTime=37,
        positiveScanEvent="None",
        negativeScanEvent="None",
        eicSmoothingWindow="None",
        eicSmoothingWindowSize=0,
        eicSmoothingPolynom=0,
        artificialMPshift_start=0,
        artificialMPshift_stop=0,
        correctCCount=True,
        minCorrelation=0.85,
        minCorrelationConnections=0.4,
        hAIntensityError=5.0,
        hAMinScans=3,
        adducts=[],
        elements=[],
        heteroAtoms=[],
        simplifyInSourceFragments=True,
        chromPeakFile=None,
        lock=None,
        queue=None,
        pID=-1,
        meVersion="NA",
        scanIndexOffset=0,
        peak_picker=None,
        peak_filter_config=None,
    ):
        # System
        self.lock = lock
        self.queue = queue
        self.pID = pID
        self.meVersion = meVersion

        self.peak_picker = peak_picker
        self.peak_filter_config = peak_filter_config
        # Check if configuredTracer is None when metabolisationExperiment is True
        if metabolisationExperiment and configuredTracer is None:
            self.printMessage("Metabolisation experiment requires a configured tracer, but none was provided.", type="error")
            raise Exception("Metabolisation experiment requires a configured tracer, but none was provided.")

        ma, ea = getIsotopeMass(configuredTracer.isotopeA if metabolisationExperiment else labellingisotopeA)
        mb, eb = getIsotopeMass(configuredTracer.isotopeB if metabolisationExperiment else labellingisotopeB)
        xOffset = mb - ma if metabolisationExperiment else xOffset

        if (not metabolisationExperiment and len(ea) > 0 and ea == eb) or metabolisationExperiment or (ea == "Hydrogen" and eb == "Deuterium"):
            pass
        else:
            self.printMessage("Labelling specifications are invalid..", type="warning")
            raise Exception("Labelling specifications are invalid..")

        if positiveScanEvent == "None" and negativeScanEvent == "None":
            self.printMessage("No scan event was specified..", type="warning")
            raise Exception("No scan event was specified..")

        self.file = file
        self.writeFeatureML = writeFeatureML
        self.writeTSV = writeTSV
        self.writeMZXML = writeMZXML

        # E. Experiment
        self.experimentOperator = exOperator
        self.experimentID = exExperimentID
        self.experimentComments = exComments
        self.experimentName = exExperimentName

        # 0. General
        self.startTime = startTime
        self.stopTime = stopTime
        self.positiveScanEvent = positiveScanEvent
        self.negativeScanEvent = negativeScanEvent

        self.metabolisationExperiment = metabolisationExperiment
        self.labellingElement = getShortName(ea)
        self.isotopeA = ma
        self.isotopeB = mb
        self.useCIsotopePatternValidation = useCIsotopePatternValidation
        self.minRatio = minRatio
        self.maxRatio = maxRatio
        self.useRatio = useRatio

        self.configuredTracer = configuredTracer
        if not self.metabolisationExperiment:
            self.configuredTracer = ConfiguredTracer(name="FML", id=0)

        # 1. Mass picking
        self.intensityThreshold = intensityThreshold
        self.intensityCutoff = intensityCutoff
        self.maxLoading = maxLoading

        self.xCounts = []
        self.xCountsString = xCounts
        a = xCounts.replace(" ", "").replace(";", ",").split(",")
        for j in a:
            if "-" in j:
                self.xCounts.extend(range(int(j[0 : j.find("-")]), int(j[j.find("-") + 1 : len(j)]) + 1))
            elif ":" in j:
                self.xCounts.extend(range(int(j[0 : j.find(":")]), int(j[j.find(":") + 1 : len(j)]) + 1))
            elif j != "":
                self.xCounts.append(int(j))
        self.xCounts = sorted(list(set(self.xCounts)))

        self.xOffset = xOffset
        self.scanIndexOffset = scanIndexOffset
        self.ppm = ppm
        self.isotopicPatternCountLeft = isotopicPatternCountLeft
        self.isotopicPatternCountRight = isotopicPatternCountRight
        self.lowAbundanceIsotopeCutoff = lowAbundanceIsotopeCutoff
        self.intensityThresholdIsotopologs = intensityThresholdIsotopologs
        self.intensityErrorN = intensityErrorN
        self.intensityErrorL = intensityErrorL
        self.purityN = purityN
        self.purityL = purityL

        # 2. Results clustering
        self.minSpectraCount = max(1, minSpectraCount)
        self.clustPPM = clustPPM

        # 3. Peak detection
        self.chromPeakPPM = chromPeakPPM

        if eicSmoothingWindow is None or eicSmoothingWindow == "" or eicSmoothingWindow == "none":
            eicSmoothingWindow = "None"
        self.eicSmoothingWindow = eicSmoothingWindow
        self.eicSmoothingWindowSize = eicSmoothingWindowSize
        self.eicSmoothingPolynom = eicSmoothingPolynom
        self.artificialMPshift_start = artificialMPshift_start
        self.artificialMPshift_stop = artificialMPshift_stop
        self.snrTh = snrTh
        self.scales = scales
        self.peakCenterError = peakCenterError
        self.peakScaleError = peakScaleError
        self.minPeakCorr = minPeakCorr
        self.checkPeaksRatio = checkPeaksRatio
        self.minPeaksRatio = minPeaksRatio
        self.maxPeaksRatio = maxPeaksRatio
        self.checkPeakWidthFilter = checkPeakWidthFilter
        self.minPeakWidth = minPeakWidth
        self.maxPeakWidth = maxPeakWidth
        self.minFWHM = minFWHM
        self.maxFWHM = maxFWHM

        self.calcIsoRatioNative = calcIsoRatioNative
        self.calcIsoRatioLabelled = calcIsoRatioLabelled
        self.calcIsoRatioMoiety = calcIsoRatioMoiety

        self.chromPeakFile = chromPeakFile  # kept for backward compatibility (no longer used)

        self.performCorrectCCount = correctCCount

        self.minCorrelation = minCorrelation
        self.minCorrelationConnections = minCorrelationConnections

        self.hAIntensityError = hAIntensityError
        self.hAMinScans = hAMinScans
        self.adducts = adducts
        self.elements = {}
        for elem in elements:
            self.elements[elem.name] = elem
        self.heteroAtoms = {}
        for heteroAtom in heteroAtoms:
            self.heteroAtoms[heteroAtom.name] = heteroAtom
        self.simplifyInSourceFragments = simplifyInSourceFragments

        self.printMessage(f"File: {file}")

        self.last_message_sent = time.time()
        self.last_value_sent = time.time()

    # Thread safe printing function
    def printMessage(self, message, type="info"):
        # Always print to stdout for debugging
        if self.lock is not None:
            self.lock.acquire()
            if type.lower() == "info":
                self.postMessageToProgressWrapper(mes="log", val=message)
            elif type.lower() == "warning":
                self.postMessageToProgressWrapper(mes="log", val=message)
            elif type.lower() == "error":
                self.postMessageToProgressWrapper(mes="log", val=message)
            else:
                self.postMessageToProgressWrapper(mes="log", val=message)
            self.lock.release()

    # helper function used to update the status of the current processing in the Process Dialog
    def postMessageToProgressWrapper(self, mes, val="", force=False):
        # Always print to stdout for debugging
        if self.pID != -1 and self.queue is not None:
            if mes.lower() == "text":
                cur = time.time()
                if cur - self.last_message_sent > 3 or force:
                    self.queue.put(Bunch(pid=self.pID, mes="text", val="%d: %s" % (self.pID, val)))
                    self.last_message_sent = cur
            elif mes == "value":
                cur = time.time()
                if cur - self.last_value_sent > 3 or force:
                    self.queue.put(Bunch(pid=self.pID, mes=mes, val=val))
                    self.last_value_sent = cur
            elif mes == "max":
                self.queue.put(Bunch(pid=self.pID, mes=mes, val=val))
                self.last_value_sent = time.time()
            elif mes == "start" or mes == "end" or mes == "failed":
                self.queue.put(Bunch(pid=self.pID, mes=mes))
                self.last_message_sent = time.time()
            elif mes == "log":
                self.queue.put(Bunch(pid=self.pID, mes=mes, val=val))
                self.last_message_sent = time.time()

    def getMostLikelyHeteroIsotope(self, foundIsotopes):
        if len(foundIsotopes) == 0:
            return

        for iso in foundIsotopes:
            isoD = foundIsotopes[iso]
            useCn = -1
            useCnRatio = 0.0
            useCnRatioTheo = 0.0

            for cn in isoD:
                cnF = isoD[cn]

                cnRatio = mean([x[1] for x in cnF])
                cnRatioTheo = mean([x[2] for x in cnF])
                len(cnF)

                if useCn == -1 or (abs(cnRatioTheo - cnRatio) < abs(useCnRatio - useCnRatioTheo)):
                    useCn = cn
                    useCnRatio = cnRatio
                    useCnRatioTheo = cnRatioTheo

            foundIsotopes[iso] = {useCn: isoD[useCn]}

    # store configuration used for processing the LC-HRMS file into the database
    def writeConfigurationToDB(self, db_con):
        # db_con is a PolarsDB object
        # Create empty tables with appropriate schemas
        db_con.create_table("config", {"key": pl.Utf8, "value": pl.Utf8})

        db_con.create_table(
            "tracerConfiguration",
            {
                "id": pl.Int64,
                "name": pl.Utf8,
                "elementCount": pl.Int64,
                "natural": pl.Utf8,
                "labelling": pl.Utf8,
                "deltaMZ": pl.Float64,
                "purityN": pl.Float64,
                "purityL": pl.Float64,
                "amountN": pl.Float64,
                "amountL": pl.Float64,
                "monoisotopicRatio": pl.Float64,
                "checkRatio": pl.Utf8,
                "lowerError": pl.Float64,
                "higherError": pl.Float64,
                "tracertype": pl.Utf8,
            },
        )

        db_con.create_table("MZs", {"id": pl.Int64, "tracer": pl.Int64, "mz": pl.Float64, "lmz": pl.Float64, "tmz": pl.Float64, "xcount": pl.Int64, "scanid": pl.Int64, "scantime": pl.Float64, "loading": pl.Int64, "intensity": pl.Float64, "intensityL": pl.Float64, "ionMode": pl.Utf8})

        db_con.create_table("MZBins", {"id": pl.Int64, "mz": pl.Float64, "ionMode": pl.Utf8})

        db_con.create_table("MZBinsKids", {"mzbinID": pl.Int64, "mzID": pl.Int64})

        db_con.create_table(
            "XICs",
            {
                "id": pl.Int64,
                "avgmz": pl.Float64,
                "xcount": pl.Int64,
                "loading": pl.Int64,
                "polarity": pl.Utf8,
                "xic": pl.Utf8,
                "xic_smoothed": pl.Utf8,
                "xic_baseline": pl.Utf8,
                "xicL": pl.Utf8,
                "xicL_smoothed": pl.Utf8,
                "xicL_baseline": pl.Utf8,
                "xicfirstiso": pl.Utf8,
                "xicLfirstiso": pl.Utf8,
                "xicLfirstisoconjugate": pl.Utf8,
                "mzs": pl.Utf8,
                "mzsL": pl.Utf8,
                "mzsfirstiso": pl.Utf8,
                "mzsLfirstiso": pl.Utf8,
                "mzsLfirstisoconjugate": pl.Utf8,
                "times": pl.Utf8,
                "scanCount": pl.Int64,
                "allPeaks": pl.Utf8,
            },
        )

        db_con.create_table("tics", {"id": pl.Int64, "polarity": pl.Utf8, "scanevent": pl.Utf8, "times": pl.Utf8, "intensities": pl.Utf8})

        db_con.create_table(
            "chromPeaks",
            {
                "id": pl.Int64,
                "tracer": pl.Int64,
                "eicID": pl.Int64,
                "NPeakCenter": pl.Int64,
                "NPeakCenterMin": pl.Float64,
                "NPeakScale": pl.Float64,
                "NSNR": pl.Float64,
                "NPeakArea": pl.Float64,
                "NPeakAbundance": pl.Float64,
                "mz": pl.Float64,
                "lmz": pl.Float64,
                "tmz": pl.Float64,
                "xcount": pl.Int64,
                "xcountId": pl.Int64,
                "LPeakCenter": pl.Int64,
                "LPeakCenterMin": pl.Float64,
                "LPeakScale": pl.Float64,
                "LSNR": pl.Float64,
                "LPeakArea": pl.Float64,
                "LPeakAbundance": pl.Float64,
                "Loading": pl.Int64,
                "peaksCorr": pl.Float64,
                "heteroAtoms": pl.Utf8,
                "NBorderLeft": pl.Int64,
                "NBorderRight": pl.Int64,
                "LBorderLeft": pl.Int64,
                "LBorderRight": pl.Int64,
                "N_startRT": pl.Float64,
                "N_endRT": pl.Float64,
                "L_startRT": pl.Float64,
                "L_endRT": pl.Float64,
                "N_FWHM": pl.Float64,
                "L_FWHM": pl.Float64,
                "N_Baseline": pl.Float64,
                "L_Baseline": pl.Float64,
                "adducts": pl.Utf8,
                "heteroAtomsFeaturePairs": pl.Utf8,
                "massSpectrumID": pl.Int64,
                "ionMode": pl.Utf8,
                "assignedMZs": pl.Utf8,
                "fDesc": pl.Utf8,
                "peaksRatio": pl.Float64,
                "peaksRatioMp1": pl.Float64,
                "peaksRatioMPm1": pl.Float64,
                "isotopesRatios": pl.Utf8,
                "mzDiffErrors": pl.Utf8,
                "isotopologRatios": pl.Utf8,
                "isoAreas": pl.Utf8,
                "isoEnrichment": pl.Utf8,
                "isoCount": pl.Int64,
                "peakType": pl.Utf8,
                "assignedName": pl.Utf8,
                "correlationsToOthers": pl.Utf8,
                "comments": pl.Utf8,
                "artificialEICLShift": pl.Int64,
            },
        )

        db_con.create_table(
            "allChromPeaks",
            {
                "id": pl.Int64,
                "tracer": pl.Int64,
                "eicID": pl.Int64,
                "NPeakCenter": pl.Int64,
                "NPeakCenterMin": pl.Float64,
                "NPeakScale": pl.Float64,
                "NSNR": pl.Float64,
                "NPeakArea": pl.Float64,
                "NPeakAbundance": pl.Float64,
                "mz": pl.Float64,
                "lmz": pl.Float64,
                "tmz": pl.Float64,
                "xcount": pl.Int64,
                "xcountId": pl.Int64,
                "LPeakCenter": pl.Int64,
                "LPeakCenterMin": pl.Float64,
                "LPeakScale": pl.Float64,
                "LSNR": pl.Float64,
                "LPeakArea": pl.Float64,
                "LPeakAbundance": pl.Float64,
                "Loading": pl.Int64,
                "peaksCorr": pl.Float64,
                "heteroAtoms": pl.Utf8,
                "NBorderLeft": pl.Int64,
                "NBorderRight": pl.Int64,
                "LBorderLeft": pl.Int64,
                "LBorderRight": pl.Int64,
                "N_startRT": pl.Float64,
                "N_endRT": pl.Float64,
                "L_startRT": pl.Float64,
                "L_endRT": pl.Float64,
                "N_FWHM": pl.Float64,
                "L_FWHM": pl.Float64,
                "N_Baseline": pl.Float64,
                "L_Baseline": pl.Float64,
                "adducts": pl.Utf8,
                "heteroAtomsFeaturePairs": pl.Utf8,
                "ionMode": pl.Utf8,
                "assignedMZs": pl.Int64,
                "fDesc": pl.Utf8,
                "peaksRatio": pl.Float64,
                "peaksRatioMp1": pl.Float64,
                "peaksRatioMPm1": pl.Float64,
                "isotopesRatios": pl.Utf8,
                "mzDiffErrors": pl.Utf8,
                "isotopologRatios": pl.Utf8,
                "isoAreas": pl.Utf8,
                "isoEnrichment": pl.Utf8,
                "isoCount": pl.Int64,
                "peakType": pl.Utf8,
                "assignedName": pl.Utf8,
                "comment": pl.Utf8,
                "comments": pl.Utf8,
                "artificialEICLShift": pl.Int64,
            },
        )

        db_con.create_table("featureGroups", {"id": pl.Int64, "featureName": pl.Utf8, "tracer": pl.Int64})

        db_con.create_table("featureGroupFeatures", {"id": pl.Int64, "fID": pl.Int64, "fDesc": pl.Utf8, "fGroupID": pl.Int64})

        db_con.create_table("featurefeatures", {"fID1": pl.Int64, "fID2": pl.Int64, "corr": pl.Float64, "silRatioValue": pl.Float64, "desc1": pl.Utf8, "desc2": pl.Utf8, "add1": pl.Utf8, "add2": pl.Utf8})

        db_con.create_table("massspectrum", {"mID": pl.Int64, "fgID": pl.Int64, "time": pl.Float64, "mzs": pl.Utf8, "intensities": pl.Utf8, "ionMode": pl.Utf8})

        db_con.create_table("stats", {"key": pl.Utf8, "value": pl.Utf8})

        db_con.insert_row("config", {"key": "MetExtractVersion", "value": self.meVersion})

        db_con.insert_row("config", {"key": "ExperimentName", "value": self.experimentName})
        db_con.insert_row("config", {"key": "ExperimentOperator", "value": self.experimentOperator})
        db_con.insert_row("config", {"key": "ExperimentID", "value": self.experimentID})
        db_con.insert_row("config", {"key": "ExperimentComments", "value": self.experimentComments})

        db_con.insert_row("config", {"key": "labellingElement", "value": self.labellingElement})
        db_con.insert_row("config", {"key": "isotopeA", "value": self.isotopeA})
        db_con.insert_row("config", {"key": "isotopeB", "value": self.isotopeB})
        db_con.insert_row("config", {"key": "useCValidation", "value": self.useCIsotopePatternValidation})
        db_con.insert_row("config", {"key": "minRatio", "value": self.minRatio})
        db_con.insert_row("config", {"key": "maxRatio", "value": self.maxRatio})
        db_con.insert_row("config", {"key": "useRatio", "value": str(self.useRatio)})
        db_con.insert_row("config", {"key": "metabolisationExperiment", "value": str(self.metabolisationExperiment)})
        db_con.insert_row("config", {"key": "configuredTracer", "value": base64.b64encode(dumps(self.configuredTracer)).decode("utf-8")})
        db_con.insert_row("config", {"key": "startTime", "value": self.startTime})
        db_con.insert_row("config", {"key": "stopTime", "value": self.stopTime})
        db_con.insert_row("config", {"key": "positiveScanEvent", "value": self.positiveScanEvent})
        db_con.insert_row("config", {"key": "negativeScanEvent", "value": self.negativeScanEvent})
        db_con.insert_row("config", {"key": "intensityThreshold", "value": self.intensityThreshold})
        db_con.insert_row("config", {"key": "intensityCutoff", "value": self.intensityCutoff})
        db_con.insert_row("config", {"key": "maxLoading", "value": self.maxLoading})
        db_con.insert_row("config", {"key": "xCounts", "value": self.xCountsString})
        db_con.insert_row("config", {"key": "xOffset", "value": self.xOffset})
        db_con.insert_row("config", {"key": "scanIndexOffset", "value": self.scanIndexOffset})
        db_con.insert_row("config", {"key": "ppm", "value": self.ppm})
        db_con.insert_row("config", {"key": "isotopicPatternCountLeft", "value": self.isotopicPatternCountLeft})
        db_con.insert_row("config", {"key": "isotopicPatternCountRight", "value": self.isotopicPatternCountRight})
        db_con.insert_row("config", {"key": "lowAbundanceIsotopeCutoff", "value": str(self.lowAbundanceIsotopeCutoff)})
        db_con.insert_row("config", {"key": "intensityThresholdIsotopologs", "value": str(self.intensityThresholdIsotopologs)})
        db_con.insert_row("config", {"key": "intensityErrorN", "value": self.intensityErrorN})
        db_con.insert_row("config", {"key": "intensityErrorL", "value": self.intensityErrorL})
        db_con.insert_row("config", {"key": "purityN", "value": self.purityN})
        db_con.insert_row("config", {"key": "purityL", "value": self.purityL})
        db_con.insert_row("config", {"key": "clustPPM", "value": self.clustPPM})
        db_con.insert_row("config", {"key": "chromPeakPPM", "value": self.chromPeakPPM})
        db_con.insert_row("config", {"key": "eicSmoothing", "value": self.eicSmoothingWindow})
        db_con.insert_row("config", {"key": "eicSmoothingWindowSize", "value": self.eicSmoothingWindowSize})
        db_con.insert_row("config", {"key": "eicSmoothingPolynom", "value": self.eicSmoothingPolynom})
        db_con.insert_row("config", {"key": "artificialMPshift_start", "value": self.artificialMPshift_start})
        db_con.insert_row("config", {"key": "artificialMPshift_stop", "value": self.artificialMPshift_stop})
        db_con.insert_row("config", {"key": "peakAbundanceCriteria", "value": "Center +- %d signals (%d total)" % (peakAbundanceUseSignalsSides, peakAbundanceUseSignals)})
        db_con.insert_row("config", {"key": "minPeakCorr", "value": self.minPeakCorr})
        db_con.insert_row("config", {"key": "checkPeaksRatio", "value": str(self.checkPeaksRatio)})
        db_con.insert_row("config", {"key": "minPeaksRatio", "value": self.minPeaksRatio})
        db_con.insert_row("config", {"key": "maxPeaksRatio", "value": self.maxPeaksRatio})
        db_con.insert_row("config", {"key": "calcIsoRatioNative", "value": self.calcIsoRatioNative})
        db_con.insert_row("config", {"key": "calcIsoRatioLabelled", "value": self.calcIsoRatioLabelled})
        db_con.insert_row("config", {"key": "calcIsoRatioMoiety", "value": self.calcIsoRatioMoiety})
        db_con.insert_row("config", {"key": "minSpectraCount", "value": self.minSpectraCount})
        db_con.insert_row("config", {"key": "configuredHeteroAtoms", "value": base64.b64encode(dumps(self.heteroAtoms)).decode("utf-8")})
        db_con.insert_row("config", {"key": "haIntensityError", "value": self.hAIntensityError})
        db_con.insert_row("config", {"key": "haMinScans", "value": self.hAMinScans})
        db_con.insert_row("config", {"key": "minCorrelation", "value": self.minCorrelation})
        db_con.insert_row("config", {"key": "minCorrelationConnections", "value": self.minCorrelationConnections})
        db_con.insert_row("config", {"key": "adducts", "value": base64.b64encode(dumps(self.adducts)).decode("utf-8")})
        db_con.insert_row("config", {"key": "elements", "value": base64.b64encode(dumps(self.elements)).decode("utf-8")})
        db_con.insert_row("config", {"key": "simplifyInSourceFragments", "value": str(self.simplifyInSourceFragments)})

        import datetime
        import uuid

        self.processingUUID = "%s_%s_%s" % (
            str(uuid.uuid1()),
            str(platform.node()),
            str(datetime.datetime.now()),
        )
        db_con.insert_row("config", {"key": "processingUUID_ext", "value": base64.b64encode(str(self.processingUUID).encode("utf-8"))})

        i = 1
        if self.metabolisationExperiment:
            tracer = self.configuredTracer
            tracer.id = i
            db_con.insert_row(
                "tracerConfiguration",
                {
                    "id": tracer.id,
                    "name": tracer.name,
                    "elementCount": tracer.elementCount,
                    "natural": tracer.isotopeA,
                    "labelling": tracer.isotopeB,
                    "deltaMZ": getIsotopeMass(tracer.isotopeB)[0] - getIsotopeMass(tracer.isotopeA)[0],
                    "purityN": tracer.enrichmentA,
                    "purityL": tracer.enrichmentB,
                    "amountN": tracer.amountA,
                    "amountL": tracer.amountB,
                    "monoisotopicRatio": tracer.monoisotopicRatio,
                    "checkRatio": str(tracer.checkRatio),
                    "lowerError": tracer.maxRelNegBias,
                    "higherError": tracer.maxRelPosBias,
                    "tracertype": tracer.tracerType,
                },
            )

        else:
            # ConfiguredTracer(name="Full metabolome labeling experiment", id=0)
            db_con.insert_row("tracerConfiguration", {"id": 0, "name": "FLE"})
        db_con.commit()

    def parseMzXMLFile(self):
        mzxml = Chromatogram()
        mzxml.parse_file(self.file, intensityCutoff=self.intensityCutoff)
        return mzxml

    # data processing step 1: searches each mass spectrum for isotope patterns of native and highly isotope enriched
    # metabolite ions. The actual calculation and processing of the data is performed in the file findIsoPairs_matchPartners.py.
    # The positive and negative ionisation modes are processed separately.
    def findSignalPairs(self, curProgress, mzxml, tracer, reportFunction=None):
        mzs = []
        posFound = 0
        negFound = 0

        checkRatio = False
        minRatio = 0.0
        maxRatio = 0.0

        if self.metabolisationExperiment:
            checkRatio = tracer.checkRatio
            minRatio = tracer.monoisotopicRatio * tracer.maxRelNegBias
            maxRatio = tracer.monoisotopicRatio * tracer.maxRelPosBias
        else:
            checkRatio = self.useRatio
            minRatio = self.minRatio
            maxRatio = self.maxRatio

        def reportFunctionHelper(curVal, text):
            reportFunction(curVal, text)

        if self.positiveScanEvent != "None":
            if self.negativeScanEvent != "None":

                def reportFunctionHelper(curVal, text):
                    reportFunction(curVal / 2, text)

            p = matchPartners(
                mzXMLData=mzxml,
                forFile=self.file,
                labellingElement=self.labellingElement,
                useCIsotopePatternValidation=self.useCIsotopePatternValidation,
                intensityThres=self.intensityThreshold,
                isotopologIntensityThres=self.intensityThresholdIsotopologs,
                maxLoading=self.maxLoading,
                xCounts=self.xCounts,
                xOffset=self.xOffset,
                ppm=self.ppm,
                intensityErrorN=self.intensityErrorN,
                intensityErrorL=self.intensityErrorL,
                purityN=tracer.enrichmentA if self.metabolisationExperiment else self.purityN,
                purityL=tracer.enrichmentB if self.metabolisationExperiment else self.purityL,
                startTime=self.startTime,
                stopTime=self.stopTime,
                filterLine=self.positiveScanEvent,
                ionMode="+",
                peakCountLeft=self.isotopicPatternCountLeft,
                peakCountRight=self.isotopicPatternCountRight,
                lowAbundanceIsotopeCutoff=self.lowAbundanceIsotopeCutoff,
                metabolisationExperiment=self.metabolisationExperiment,
                checkRatio=checkRatio,
                minRatio=minRatio,
                maxRatio=maxRatio,
                scanIndexOffset=self.scanIndexOffset,
                reportFunction=reportFunctionHelper,
            )
            posFound = len(p)
            mzs.extend(p)

        def reportFunctionHelper(curVal, text):
            reportFunction(curVal, text)

        if self.negativeScanEvent != "None":
            if self.positiveScanEvent != "None":

                def reportFunctionHelper(curVal, text):
                    reportFunction(0.5 + curVal / 2, text)

            n = matchPartners(
                mzXMLData=mzxml,
                forFile=self.file,
                labellingElement=self.labellingElement,
                useCIsotopePatternValidation=self.useCIsotopePatternValidation,
                intensityThres=self.intensityThreshold,
                isotopologIntensityThres=self.intensityThresholdIsotopologs,
                maxLoading=self.maxLoading,
                xCounts=self.xCounts,
                xOffset=self.xOffset,
                ppm=self.ppm,
                intensityErrorN=self.intensityErrorN,
                intensityErrorL=self.intensityErrorL,
                purityN=tracer.enrichmentA if self.metabolisationExperiment > 1 else self.purityN,
                purityL=tracer.enrichmentB if self.metabolisationExperiment > 1 else self.purityL,
                startTime=self.startTime,
                stopTime=self.stopTime,
                filterLine=self.negativeScanEvent,
                ionMode="-",
                peakCountLeft=self.isotopicPatternCountLeft,
                peakCountRight=self.isotopicPatternCountRight,
                lowAbundanceIsotopeCutoff=self.lowAbundanceIsotopeCutoff,
                metabolisationExperiment=self.metabolisationExperiment,
                checkRatio=checkRatio,
                minRatio=minRatio,
                maxRatio=maxRatio,
                scanIndexOffset=self.scanIndexOffset,
                reportFunction=reportFunctionHelper,
            )
            negFound = len(n)
            mzs.extend(n)

        return mzs, negFound, posFound

    # store detected signal pairs (1st data processing step) in the database
    def writeSignalPairsToDB(self, mzs, mzxml, tracerID):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        pdata = {
            "id": [],
            "tracer": [],
            "mz": [],
            "lmz": [],
            "tmz": [],
            "xcount": [],
            "scanid": [],
            "scantime": [],
            "loading": [],
            "intensity": [],
            "intensityL": [],
            "ionMode": [],
        }

        for mz in mzs:
            mz.id = self.curMZId
            mz.tid = tracerID

            pdata["id"].append(mz.id)
            pdata["tracer"].append(mz.tid)
            pdata["mz"].append(mz.mz)
            pdata["lmz"].append(mz.lmz)
            pdata["tmz"].append(mz.tmz)
            pdata["xcount"].append(mz.xCount)
            pdata["scanid"].append(mz.scanid)
            pdata["scantime"].append(mz.scantime)
            pdata["loading"].append(mz.loading)
            pdata["intensity"].append(mz.nIntensity)
            pdata["intensityL"].append(mz.lIntensity)
            pdata["ionMode"].append(mz.ionMode)

            self.curMZId = self.curMZId + 1

        new_mzs_df = pl.DataFrame(pdata)
        existing_mzs_df = db_con.tables.get("MZs", pl.DataFrame())
        if len(existing_mzs_df) > 0 and len(new_mzs_df) > 0:
            db_con.set_table("MZs", pl.concat([existing_mzs_df, new_mzs_df], how="vertical_relaxed"))
        elif len(new_mzs_df) > 0:
            db_con.set_table("MZs", new_mzs_df)

        db_con.commit()
        db_con.close()

    # data processing step 2: cluster detected signal pairs with H
    def clusterFeaturePairs(self, mzs, reportFunction=None):
        mzbins = {}
        mzbins["+"] = []
        mzbins["-"] = []

        uniquexCounts = list(set([mz.xCount for mz in mzs]))

        # cluster each detected number of carbon atoms separately
        donei = 0
        for xCount in uniquexCounts:
            if reportFunction is not None:
                reportFunction(donei / len(uniquexCounts), "Current Xn: %s" % xCount)
            # cluster each detected number of loadings separately
            for loading in range(self.maxLoading, 0, -1):
                for ionMode in ["+", "-"]:
                    xAct = sorted(
                        [mz for mz in mzs if mz.ionMode == ionMode and mz.xCount == xCount and mz.loading == loading],
                        key=lambda x: x.mz,
                    )

                    doClusterings = []

                    if len(xAct) > 0:
                        lastUnused = 0
                        usedCount = 0

                        # non-consecutive mz regions are prematurely separated without HCA to reduce the number of
                        # signal pairs for HCA (faster processing)
                        for i in range(1, len(xAct)):
                            ppmDiff = (xAct[i].mz - xAct[i - 1].mz) * 1000000 / xAct[i - 1].mz
                            if ppmDiff > self.clustPPM:
                                doClusterings.append(xAct[lastUnused:i])

                                usedCount = usedCount + i - lastUnused
                                lastUnused = i

                        # cluster remainign signal pairs
                        if lastUnused < len(xAct):
                            doClusterings.append(xAct[lastUnused : len(xAct)])
                            usedCount = usedCount + len(xAct) - lastUnused

                        assert usedCount == len(xAct)

                        temp = doClusterings
                        doClusterings = []
                        for t in temp:
                            xAct = sorted([mz for mz in t], key=lambda x: x.scanIndex)

                            lastUnused = 0
                            usedCount = 0

                            for i in range(1, len(xAct)):
                                scanDiff = xAct[i].scanIndex - xAct[i - 1].scanIndex
                                if scanDiff > 20:
                                    doClusterings.append(xAct[lastUnused:i])

                                    usedCount = usedCount + i - lastUnused
                                    lastUnused = i

                            # cluster remainign signal pairs
                            if lastUnused < len(xAct):
                                doClusterings.append(xAct[lastUnused : len(xAct)])
                                usedCount = usedCount + len(xAct) - lastUnused

                            assert usedCount == len(xAct)

                        for doClustering in doClusterings:
                            hc = HierarchicalClustering(
                                doClustering,
                                dist=lambda x, y: x.getValue() - y.getValue(),
                                val=lambda x: x.mz,
                                mean=lambda x, y: x / y,
                                add=lambda x, y: x + y,
                            )

                            for n in cutTreeSized(hc.getTree(), self.clustPPM):
                                mzbins[ionMode].append(n)

        return mzbins

    # store signal pair clusters (2nd data processing step) in the database
    def writeFeaturePairClustersToDB(self, mzbins):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        mzbins_data = {"id": [], "mz": [], "ionMode": []}
        mzbinskids = {"mzbinID": [], "mzID": []}

        for ionMode in ["+", "-"]:
            if ionMode in mzbins.keys():
                for mzbin in mzbins[ionMode]:
                    mzbins_data["id"].append(self.curMZBinId)
                    mzbins_data["mz"].append(mzbin.getValue())
                    mzbins_data["ionMode"].append(ionMode)
                    for kid in mzbin.getKids():
                        mzbinskids["mzbinID"].append(self.curMZBinId)
                        mzbinskids["mzID"].append(kid.getObject().id)
                    self.curMZBinId = self.curMZBinId + 1

        new_mzbins_df = pl.DataFrame(mzbins_data)
        new_mzbinskids_df = pl.DataFrame(mzbinskids)

        existing_mzbins_df = db_con.tables.get("MZBins", pl.DataFrame())
        if len(existing_mzbins_df) > 0 and len(new_mzbins_df) > 0:
            db_con.set_table("MZBins", pl.concat([existing_mzbins_df, new_mzbins_df], how="vertical_relaxed"))
        elif len(new_mzbins_df) > 0:
            db_con.set_table("MZBins", new_mzbins_df)

        existing_mzbinskids_df = db_con.tables.get("MZBinsKids", pl.DataFrame())
        if len(existing_mzbinskids_df) > 0 and len(new_mzbinskids_df) > 0:
            db_con.set_table("MZBinsKids", pl.concat([existing_mzbinskids_df, new_mzbinskids_df], how="vertical_relaxed"))
        elif len(new_mzbinskids_df) > 0:
            db_con.set_table("MZBinsKids", new_mzbinskids_df)

        db_con.commit()
        db_con.close()

    def removeImpossibleFeaturePairClusters(self, mzbins):
        mzBinsNew = {}
        for ionMode in mzbins.keys():
            mzBinsNew[ionMode] = [mzbin for mzbin in mzbins[ionMode] if len(mzbin.getKids()) >= self.minSpectraCount]
        return mzBinsNew

    # EXPERIMENTAL: calulcate the mean intensity ratio of two chromatographic peaks at the same retention time
    # in two different EICs. Thus, the internal standardisation is not calculated using the peak area
    # but rather the ratio of each MS peak that contributes to the chromatographic peak.
    def getMeanRatioOfScans(self, eicA, eicB, lib, rib, perfWeighted=True, minInt=1000, minRatiosNecessary=3):
        try:
            if perfWeighted:
                sumEIC = sum(eicA[lib:rib])
                sumEICL = sum(eicB[lib:rib])

                normEIC = eicA
                if sumEICL > sumEIC:
                    normEIC = eicB

                os = [o for o in range(lib, rib) if eicA[o] >= minInt and eicB[o] >= minInt and eicB[o] > 0]
                normSum = sum([normEIC[o] for o in os])
                weights = [1.0 * normEIC[o] / normSum for o in os]
                ratios = [1.0 * eicA[o] / eicB[o] for o in os if eicB[o] > minInt and eicB[o] > 0]

                if len(ratios) >= minRatiosNecessary:
                    assert len(ratios) == len(weights)
                    ratio = sum([ratios[o] * weights[o] for o in range(len(ratios))])
                    sum([ratios[o] * weights[o] for o in range(len(ratios))])
                    return ratio
                else:
                    return -1
            else:
                ratios = [(eicA[o] / eicB[o]) for o in range(lib, rib) if eicA[o] >= minInt and eicB[o] >= minInt]
                if len(ratios) >= minRatiosNecessary:
                    return mean(ratios)
                else:
                    return -1

        except IndexError:
            return -1

    # data processing step 3: for each signal pair cluster extract the EICs, detect chromatographic peaks
    # present in both EICs at approximately the same retention time and very their chromatographic peak shapes.
    # if all criteria are passed, write this detected feature pair to the database
    def findChromatographicPeaksAndWriteToDB(self, mzbins, mzxml, tracerID, reportFunction=None):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())
        chromPeaks = []

        totalBins = len(mzbins["+"]) + len(mzbins["-"])

        # process each signal pair cluster
        for ionMode in ["+", "-"]:
            if ionMode == "+":
                scanEvent = self.positiveScanEvent
            elif ionMode == "-":
                scanEvent = self.negativeScanEvent
            else:
                raise Exception("Unknown Ion Mode")

            if scanEvent != "None":
                for mzbin, i in zip(mzbins[ionMode], range(0, len(mzbins[ionMode]))):
                    try:
                        if reportFunction is not None:
                            doneBins = i
                            if ionMode == "-":
                                doneBins += len(mzbins["+"])

                            reportFunction(
                                1.0 * doneBins / totalBins,
                                "%d mzbins remaining" % (totalBins - doneBins),
                            )

                        kids = mzbin.getKids()
                        if len(kids) < self.minSpectraCount:
                            continue

                        # calulcate mean mz value for this signal pair cluster
                        meanmz = weightedMean(
                            [kid.getObject().mz for kid in kids],
                            [kid.getObject().nIntensity for kid in kids],
                        )
                        meanmzLabelled = weightedMean(
                            [kid.getObject().lmz for kid in kids],
                            [kid.getObject().lIntensity for kid in kids],
                        )
                        meantmz = weightedMean(
                            [kid.getObject().tmz for kid in kids],
                            [kid.getObject().lIntensity for kid in kids],
                        )

                        xcount = kids[0].getObject().xCount
                        assert all([kid.getObject().xCount == xcount for kid in kids])

                        loading = kids[0].getObject().loading
                        assert all([kid.getObject().loading == loading for kid in kids])

                        # extract the EIC of the native ion and detect its chromatographic peaks
                        # optionally: smoothing
                        eic, times, scanIds, mzs = mzxml.getEIC(meanmz, self.chromPeakPPM, filterLine=scanEvent)
                        eicBaseline = self.BL.getBaseline(copy(eic), times)
                        eicSmoothed = smoothDataSeries(
                            times,
                            copy(eic),
                            windowLen=self.eicSmoothingWindowSize,
                            window=self.eicSmoothingWindow,
                            polynom=self.eicSmoothingPolynom,
                        )
                        # extract the EIC of the labelled ion and detect its chromatographic peaks
                        # optionally: smoothing
                        eicL, times, scanIds, mzsL = mzxml.getEIC(meanmzLabelled, self.chromPeakPPM, filterLine=scanEvent)
                        eicLBaseline = self.BL.getBaseline(copy(eicL), times)
                        eicLSmoothed = smoothDataSeries(
                            times,
                            copy(eicL),
                            windowLen=self.eicSmoothingWindowSize,
                            window=self.eicSmoothingWindow,
                            polynom=self.eicSmoothingPolynom,
                        )

                        ## TODO needs to be optimized
                        # startIndex=max(0, minInd-int(ceil(self.scales[1]*10)))
                        # endIndex=min(len(eic)-1, int(ceil(maxInd+self.scales[1]*10)))
                        startIndex = 0
                        endIndex = len(eic) - 1

                        # Use unified peak picker and filter config
                        peaksN = self.peak_picker.getPeaksFor(times, eicSmoothed, startIndex=startIndex, endIndex=endIndex)
                        peaksL = self.peak_picker.getPeaksFor(times, eicLSmoothed, startIndex=startIndex, endIndex=endIndex)

                        if self.peak_filter_config is not None:
                            peaksN = filter_peaks(peaksN, config=self.peak_filter_config)
                            peaksL = filter_peaks(peaksL, config=self.peak_filter_config)

                        # Recalculate SNR, FWHM, area from raw (unsmoothed) EICs
                        # Peak boundaries are kept from smoothed-EIC detection, only metrics are recomputed
                        times_arr = np.asarray(times)
                        eic_raw_arr = np.asarray(eic)
                        eicL_raw_arr = np.asarray(eicL)
                        for pk in peaksN:
                            pk.snr = BasePeakPicker.compute_snr(eic_raw_arr, pk.apex_index, pk.start_index, pk.end_index)
                            pk.fwhm = BasePeakPicker.compute_fwhm(times_arr, eic_raw_arr, pk.apex_index, pk.start_index, pk.end_index)
                            pk.area = BasePeakPicker.compute_area(times_arr, eic_raw_arr, pk.start_index, pk.end_index)
                            pk.apex_intensity = float(eic_raw_arr[pk.apex_index])
                        for pk in peaksL:
                            pk.snr = BasePeakPicker.compute_snr(eicL_raw_arr, pk.apex_index, pk.start_index, pk.end_index)
                            pk.fwhm = BasePeakPicker.compute_fwhm(times_arr, eicL_raw_arr, pk.apex_index, pk.start_index, pk.end_index)
                            pk.area = BasePeakPicker.compute_area(times_arr, eicL_raw_arr, pk.start_index, pk.end_index)
                            pk.apex_intensity = float(eicL_raw_arr[pk.apex_index])

                        # get EICs of M+1, M'-1 and M'+1 for the database
                        eicfirstiso, timesL, scanIdsL, mzsfirstiso = mzxml.getEIC(
                            meanmz + 1.00335484 / loading,
                            self.chromPeakPPM,
                            filterLine=scanEvent,
                        )
                        # eicfirstisoSmoothed = smoothDataSeries(times, eicfirstiso, windowLen=self.eicSmoothingWindowSize, window=self.eicSmoothingWindow, polynom=self.eicSmoothingPolynom)

                        eicLfirstiso, timesL, scanIdsL, mzsLfirstiso = mzxml.getEIC(
                            meanmzLabelled - 1.00335484 / loading,
                            self.chromPeakPPM,
                            filterLine=scanEvent,
                        )
                        # eicLfirstisoSmoothed = smoothDataSeries(times, eicLfirstiso, windowLen=self.eicSmoothingWindowSize, window=self.eicSmoothingWindow, polynom=self.eicSmoothingPolynom)

                        eicLfirstisoconjugate, timesL, scanIdsL, mzsLfirstisoconjugate = mzxml.getEIC(
                            meanmzLabelled + 1.00335484 / loading,
                            self.chromPeakPPM,
                            filterLine=scanEvent,
                        )
                        # eicLfirstisoconjugateSmoothed = smoothDataSeries(times, eicLfirstisoconjugate, windowLen=self.eicSmoothingWindowSize, window=self.eicSmoothingWindow, polynom=self.eicSmoothingPolynom)

                        peaksBoth = []

                        # match detected chromatographic peaks from both EICs
                        for peakN in peaksN:
                            closestMatch = -1
                            closestOffset = 10000000000

                            # search closest peak pair
                            for li, peakL in enumerate(peaksL):
                                if abs(peakN.apex_index - peakL.apex_index) <= self.peakCenterError:
                                    if closestMatch == -1 or closestOffset > abs(peakN.apex_index - peakL.apex_index):
                                        closestMatch = li
                                        closestOffset = abs(peakN.apex_index - peakL.apex_index)

                            if closestMatch != -1:
                                peakL = peaksL[closestMatch]

                                # print("M: ", int(peakN.peakIndex - peakN.peakLeftFlank), peakN.peakIndex, int(peakN.peakIndex + peakN.peakRightFlank))
                                # print(f"   FWHM: {fwhm_M}, area: {area_M}, SNR: {snr_M}")
                                # print("Mp: ", int(peakL.peakIndex - peakL.peakLeftFlank), peakL.peakIndex, int(peakL.peakIndex + peakL.peakRightFlank))
                                # print(f"   FWHM: {fwhm_Mp}, area: {area_Mp}, SNR: {snr_Mp}")

                                peak = ChromPeakPair(
                                    mz=meanmz,
                                    lmz=meanmzLabelled,
                                    tmz=meantmz,
                                    xCount=xcount,
                                    loading=loading,
                                    ionMode=ionMode,
                                    NPeakCenter=peakN.apex_index,
                                    NPeakCenterMin=peakN.apex_rt,
                                    NPeakScale=(peakN.apex_index - peakN.start_index + peakN.end_index - peakN.apex_index) / 2.0,
                                    NSNR=peakN.snr,
                                    NPeakArea=peakN.area,
                                    LPeakCenter=peakL.apex_index,
                                    LPeakCenterMin=peakL.apex_rt,
                                    LPeakScale=(peakL.apex_index - peakL.start_index + peakL.end_index - peakL.apex_index) / 2.0,
                                    LSNR=peakL.snr,
                                    LPeakArea=peakL.area,
                                    NXIC=eic,
                                    LXIC=eicL,
                                    times=times,
                                    fDesc=[],
                                    adducts=[],
                                    heteroAtomsFeaturePairs=[],
                                    NXICSmoothed=eicSmoothed,
                                    LXICSmoothed=eicLSmoothed,
                                    NBorderLeft=peakN.apex_index - peakN.start_index,
                                    NBorderRight=peakN.end_index - peakN.apex_index,
                                    LBorderLeft=peakL.apex_index - peakL.start_index,
                                    LBorderRight=peakL.end_index - peakL.apex_index,
                                    N_startRT=peakN.start_rt,
                                    N_endRT=peakN.end_rt,
                                    L_startRT=peakL.start_rt,
                                    L_endRT=peakL.end_rt,
                                    N_FWHM=peakN.fwhm,
                                    L_FWHM=peakL.fwhm,
                                    N_Baseline=peakN.baseline,
                                    L_Baseline=peakL.baseline,
                                    isotopeRatios=[],
                                    mzDiffErrors=Bunch(),
                                    comments=[],
                                    artificialEICLShift=0,
                                )

                                lb = int(
                                    max(
                                        0,
                                        min(
                                            peak.NPeakCenter - peak.NBorderLeft,
                                            peak.LPeakCenter - peak.LBorderLeft,
                                        ),
                                    )
                                )
                                rb = int(
                                    min(
                                        max(
                                            peak.NPeakCenter + peak.NBorderRight,
                                            peak.LPeakCenter + peak.LBorderRight,
                                        )
                                        + 1,
                                        len(eic) - 1,
                                    )
                                )
                                peakN = eicSmoothed[lb:rb]
                                peakL = eicLSmoothed[lb:rb]

                                def findBestArtificialShift(eicN, eicL, lb, rb, shiftFrom=0, shiftTo=0):
                                    correlations = []

                                    for artShift in range(shiftFrom, shiftTo + 1):
                                        peakN = eicN[lb:rb]
                                        peakL = eicL[(lb + artShift) : (rb + artShift)]
                                        silRatios = [
                                            peakN[i] / peakL[i]
                                            for i in range(
                                                int(len(peakN) * 0.25),
                                                int(len(peakN) * 0.75) + 1,
                                            )
                                            if i < len(peakL) and peakL[i] > 0 and peakN[i] > 0
                                        ]
                                        try:
                                            correlations.append(
                                                Bunch(
                                                    correlation=corr(peakN, peakL),
                                                    artificialShift=artShift,
                                                    silRatios=silRatios,
                                                    peakNInts=[
                                                        peakN[i]
                                                        for i in range(
                                                            int(len(peakN) * 0.25),
                                                            int(len(peakN) * 0.75) + 1,
                                                        )
                                                        if i < len(peakL) and peakL[i] > 0 and peakN[i] > 0
                                                    ],
                                                    peakLInts=[
                                                        peakL[i]
                                                        for i in range(
                                                            int(len(peakN) * 0.25),
                                                            int(len(peakN) * 0.75) + 1,
                                                        )
                                                        if i < len(peakL) and peakL[i] > 0 and peakN[i] > 0
                                                    ],
                                                )
                                            )
                                        except AssertionError:
                                            print(
                                                f"Warning: Could not calculate correlation for peak at RT {times[peak.NPeakCenter]} with artShift {artShift}. This is likely due to too close proximity of the EIC with its the start/end. This is a bug of the software and will be resolved in a future version"
                                            )
                                    if not correlations:
                                        return None
                                    bestFit = max(correlations, key=lambda x: x.correlation)

                                    return bestFit

                                co = findBestArtificialShift(
                                    eicSmoothed,
                                    eicLSmoothed,
                                    lb,
                                    rb,
                                    self.artificialMPshift_start,
                                    self.artificialMPshift_stop,
                                )

                                # check peak shape (Pearson correlation)

                                if co is not None and co.correlation >= self.minPeakCorr and ((not self.checkPeaksRatio) or self.minPeaksRatio <= (peak.NPeakArea / peak.LPeakArea) <= self.maxPeaksRatio):
                                    peak.peaksCorr = co.correlation
                                    peak.silRatios = Bunch(
                                        silRatios=co.silRatios,
                                        peakNInts=co.peakNInts,
                                        peakLInts=co.peakLInts,
                                    )
                                    if co.artificialShift != 0:
                                        peak.artificialEICLShift = co.artificialShift

                                    peaksBoth.append(peak)

                                    libs = int(
                                        max(
                                            peak.NPeakCenter - floor(peak.NBorderLeft * 0.33),
                                            peak.LPeakCenter - floor(peak.LBorderLeft * 0.33),
                                        )
                                    )
                                    ribs = (
                                        int(
                                            min(
                                                peak.NPeakCenter + floor(peak.NBorderRight * 0.33),
                                                peak.LPeakCenter + floor(peak.LBorderRight * 0.33),
                                            )
                                        )
                                        + 1
                                    )

                                    peak.peaksRatio = self.getMeanRatioOfScans(
                                        eic,
                                        eicL,
                                        libs,
                                        ribs,
                                        minInt=self.intensityThreshold,
                                    )
                                    peak.peaksRatioMp1 = self.getMeanRatioOfScans(
                                        eicfirstiso,
                                        eic,
                                        libs,
                                        ribs,
                                        minInt=self.intensityThreshold,
                                    )
                                    peak.peaksRatioMPm1 = self.getMeanRatioOfScans(
                                        eicLfirstiso,
                                        eicL,
                                        libs,
                                        ribs,
                                        minInt=self.intensityThreshold,
                                    )

                        # check, if enough MS scans fulfill the requirements (isotope patterns)
                        for kid in kids:
                            kido = kid.getObject()

                            closestPeak = -1
                            distance = len(eic)
                            i = 0
                            for peak in peaksBoth:
                                if abs(kido.scanIndex - peak.NPeakCenter) < distance:
                                    closestPeak = i
                                    distance = abs(kido.scanIndex - peak.NPeakCenter)
                                i = i + 1
                            if closestPeak != -1 and distance < (
                                min(
                                    peaksBoth[closestPeak].NBorderLeft,
                                    peaksBoth[closestPeak].NBorderRight,
                                )
                                * 2
                            ):
                                peaksBoth[closestPeak].assignedMZs.append(kido)

                        curChromPeaks = []
                        for peak in peaksBoth:
                            if len(peak.assignedMZs) >= self.minSpectraCount:
                                curChromPeaks.append(peak)

                        for peak in curChromPeaks:
                            lb = int(
                                max(
                                    0,
                                    min(
                                        peak.NPeakCenter - peakAbundanceUseSignalsSides,
                                        peak.LPeakCenter - peakAbundanceUseSignalsSides,
                                    ),
                                )
                            )
                            rb = int(
                                min(
                                    max(
                                        peak.NPeakCenter + peakAbundanceUseSignalsSides,
                                        peak.LPeakCenter + peakAbundanceUseSignalsSides,
                                    )
                                    + 1,
                                    len(eic) - 1,
                                )
                            )
                            peakN = eic[lb:rb]
                            peakL = eicL[lb:rb]

                            peak.NPeakAbundance = max(peakN)  # mean(peakN)
                            peak.LPeakAbundance = max(peakL)  # mean(peakL)

                        # write detected feature pair to the database
                        if len(curChromPeaks) > 0:
                            assert len(eic) == len(eicL) == len(eicfirstiso) == len(eicLfirstiso) == len(eicLfirstisoconjugate) == len(times)

                            keepScans = set()
                            for cs in range(len(eic)):
                                if eic[cs] > 0 or eicL[cs] > 0 or eicfirstiso[cs] > 0 or eicLfirstiso[cs] > 0 or eicLfirstisoconjugate[cs] > 0:
                                    keepScans.add(cs)
                            keepScans.add(0)
                            keepScans.add(len(eic) - 1)

                            # save the native and labelled EICs to the database
                            db_con.insert_row(
                                "XICs",
                                {
                                    "id": self.curEICId,
                                    "avgmz": meanmz,
                                    "xcount": xcount,
                                    "loading": kids[0].getObject().loading,
                                    "polarity": ionMode,
                                    "xic": ";".join(["%f" % i for i in eic]),
                                    "xicL": ";".join(["%f" % i for i in eicL]),
                                    "xicfirstiso": ";".join(["%f" % i for i in eicfirstiso]),
                                    "xicLfirstiso": ";".join(["%f" % i for i in eicLfirstiso]),
                                    "xicLfirstisoconjugate": ";".join(["%f" % i for i in eicLfirstisoconjugate]),
                                    "xic_smoothed": ";".join(["%f" % i for i in eicSmoothed]),
                                    "xicL_smoothed": ";".join(["%f" % i for i in eicLSmoothed]),
                                    "xic_baseline": ";".join(["%f" % i for i in eicBaseline]),
                                    "xicL_baseline": ";".join(["%f" % i for i in eicLBaseline]),
                                    "mzs": ";".join(["%f" % (mzs[i]) for i in range(0, len(eic))]),
                                    "mzsL": ";".join(["%f" % (mzsL[i]) for i in range(0, len(eic))]),
                                    "mzsfirstiso": ";".join(["%f" % (mzsfirstiso[i]) for i in range(0, len(eic))]),
                                    "mzsLfirstiso": ";".join(["%f" % (mzsLfirstiso[i]) for i in range(0, len(eic))]),
                                    "mzsLfirstisoconjugate": ";".join(["%f" % (mzsLfirstisoconjugate[i]) for i in range(0, len(eic))]),
                                    "times": ";".join(["%f" % (times[i]) for i in range(0, len(times))]),
                                    "scanCount": len(eic),
                                    "allPeaks": base64.b64encode(dumps({"peaksN": peaksN, "peaksL": peaksL})).decode("utf-8"),
                                },
                            )

                            # save the detected feature pairs in these EICs to the database
                            for peak in curChromPeaks:
                                peak.id = self.curPeakId
                                peak.eicID = self.curEICId
                                adjcCount = peak.xCount
                                peak.correctedXCount = peak.xCount

                                if self.performCorrectCCount and False:
                                    adjcCount = adjcCount + getAtomAdd(self.purityL, peak.xCount) + getAtomAdd(self.purityN, peak.xCount)
                                    peak.correctedXCount = adjcCount

                                db_con.insert_row(
                                    "chromPeaks",
                                    {
                                        "id": peak.id,
                                        "tracer": tracerID,
                                        "eicID": peak.eicID,
                                        "mz": peak.mz,
                                        "lmz": peak.lmz,
                                        "tmz": peak.tmz,
                                        "xcount": peak.correctedXCount,
                                        "xcountId": peak.xCount,
                                        "Loading": peak.loading,
                                        "ionMode": ionMode,
                                        "NPeakCenter": peak.NPeakCenter,
                                        "NPeakCenterMin": peak.NPeakCenterMin,
                                        "NPeakScale": peak.NPeakScale,
                                        "NSNR": peak.NSNR,
                                        "NPeakArea": peak.NPeakArea,
                                        "NPeakAbundance": peak.NPeakAbundance,
                                        "NBorderLeft": peak.NBorderLeft,
                                        "NBorderRight": peak.NBorderRight,
                                        "LPeakCenter": peak.LPeakCenter,
                                        "LPeakCenterMin": peak.LPeakCenterMin,
                                        "LPeakScale": peak.LPeakScale,
                                        "LSNR": peak.LSNR,
                                        "LPeakArea": peak.LPeakArea,
                                        "LPeakAbundance": peak.LPeakAbundance,
                                        "LBorderLeft": peak.LBorderLeft,
                                        "LBorderRight": peak.LBorderRight,
                                        "N_startRT": peak.N_startRT,
                                        "N_endRT": peak.N_endRT,
                                        "L_startRT": peak.L_startRT,
                                        "L_endRT": peak.L_endRT,
                                        "N_FWHM": getattr(peak, "N_FWHM", 0.0),
                                        "L_FWHM": getattr(peak, "L_FWHM", 0.0),
                                        "N_Baseline": getattr(peak, "N_Baseline", 0.0),
                                        "L_Baseline": getattr(peak, "L_Baseline", 0.0),
                                        "peaksCorr": peak.peaksCorr,
                                        "heteroAtoms": "",
                                        "adducts": "",
                                        "heteroAtomsFeaturePairs": "",
                                        "massSpectrumID": 0,
                                        "assignedMZs": base64.b64encode(dumps(peak.assignedMZs)).decode("utf-8"),
                                        "fDesc": base64.b64encode(dumps([])).decode("utf-8"),
                                        "peaksRatio": peak.peaksRatio,
                                        "peaksRatioMp1": peak.peaksRatioMp1,
                                        "peaksRatioMPm1": peak.peaksRatioMPm1,
                                        "peakType": "patternFound",
                                        "comments": base64.b64encode(dumps(peak.comments)).decode("utf-8"),
                                        "artificialEICLShift": peak.artificialEICLShift,
                                    },
                                )

                                db_con.insert_row(
                                    "allChromPeaks",
                                    {
                                        "id": peak.id,
                                        "tracer": tracerID,
                                        "eicID": peak.eicID,
                                        "mz": peak.mz,
                                        "lmz": peak.lmz,
                                        "tmz": peak.tmz,
                                        "xcount": peak.correctedXCount,
                                        "xcountId": peak.xCount,
                                        "Loading": peak.loading,
                                        "ionMode": ionMode,
                                        "NPeakCenter": peak.NPeakCenter,
                                        "NPeakCenterMin": peak.NPeakCenterMin,
                                        "NPeakScale": peak.NPeakScale,
                                        "NSNR": peak.NSNR,
                                        "NPeakArea": peak.NPeakArea,
                                        "NPeakAbundance": peak.NPeakAbundance,
                                        "NBorderLeft": peak.NBorderLeft,
                                        "NBorderRight": peak.NBorderRight,
                                        "LPeakCenter": peak.LPeakCenter,
                                        "LPeakCenterMin": peak.LPeakCenterMin,
                                        "LPeakScale": peak.LPeakScale,
                                        "LSNR": peak.LSNR,
                                        "LPeakArea": peak.LPeakArea,
                                        "LPeakAbundance": peak.LPeakAbundance,
                                        "LBorderLeft": peak.LBorderLeft,
                                        "LBorderRight": peak.LBorderRight,
                                        "N_startRT": peak.N_startRT,
                                        "N_endRT": peak.N_endRT,
                                        "L_startRT": peak.L_startRT,
                                        "L_endRT": peak.L_endRT,
                                        "N_FWHM": getattr(peak, "N_FWHM", 0.0),
                                        "L_FWHM": getattr(peak, "L_FWHM", 0.0),
                                        "N_Baseline": getattr(peak, "N_Baseline", 0.0),
                                        "L_Baseline": getattr(peak, "L_Baseline", 0.0),
                                        "peaksCorr": peak.peaksCorr,
                                        "heteroAtoms": "",
                                        "adducts": "",
                                        "heteroAtomsFeaturePairs": "",
                                        "assignedMZs": len(peak.assignedMZs),
                                        "fDesc": base64.b64encode(dumps([])).decode("utf-8"),
                                        "peaksRatio": peak.peaksRatio,
                                        "peaksRatioMp1": peak.peaksRatioMp1,
                                        "peaksRatioMPm1": peak.peaksRatioMPm1,
                                        "peakType": "patternFound",
                                        "comments": base64.b64encode(dumps(peak.comments)).decode("utf-8"),
                                        "artificialEICLShift": peak.artificialEICLShift,
                                    },
                                )

                                chromPeaks.append(peak)
                                self.curPeakId = self.curPeakId + 1
                    except Exception as e:
                        print("Error processing mzbin with meanmz %f: %s" % (meanmz, str(e)))
                        traceback.print_exc()

                    self.curEICId = self.curEICId + 1
        db_con.commit()
        db_con.close()
        return chromPeaks

    # data processing step 4: remove those feature pairs, which represent incorrect pairings of
    # isotoplogs of either the native or the labelled or both ions. Such incorrect pairings always have
    # and increased mz value and/or a decreased number of labelled carbon atoms. Such identified
    # incorrect pairings are then removed from the database and thus the processing results
    def removeFalsePositiveFeaturePairsAndUpdateDB(self, chromPeaks, reportFunction=None):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        todel = {}

        # iterate over all detected feature pairs and compare those
        # if the a) originate from the same ionisation mode
        #    and b) if they are not the same feature pair
        for a in range(0, len(chromPeaks)):
            if reportFunction is not None:
                reportFunction(
                    1.0 * a / len(chromPeaks),
                    "%d feature pairs remaining" % (len(chromPeaks) - a),
                )

            peakA = chromPeaks[a]
            for b in range(0, len(chromPeaks)):
                peakB = chromPeaks[b]
                if a != b and peakA.ionMode == peakB.ionMode and (peakA.mz < peakB.mz or abs(peakA.mz - peakB.mz) <= (peakA.mz * 2.5 * self.ppm / 1000000.0)):
                    # not same feature pair but same ionisation mode

                    if abs(peakA.NPeakCenter - peakB.NPeakCenter) <= self.peakCenterError:
                        # same retention time
                        if abs(peakA.mz - peakB.mz) <= (peakA.mz * 2.5 * self.ppm / 1000000.0):
                            # same mz value
                            if isinstance(peakA.xCount, float) and (peakB.xCount - peakA.xCount) == 2 and peakB.loading == peakA.loading and ((peakB.LPeakArea / peakA.LPeakArea) <= 0.1):
                                # incorrect matching of 18O atoms detected
                                # e.g. a and b have 869.4153; a has C39 and b has C41; peaksRatio(a)=1.46, peaksRatio(b)=43.48
                                # --> 1.46/43.48~0.0335 (which is in good agreement with On) and thus b has to be removed
                                if b not in todel.keys():
                                    todel[b] = []
                                todel[b].append("incorrectly matched hetero atoms (most likely O) with " + str(peakA.mz) + " " + str(peakA.xCount))
                            elif isinstance(peakA.xCount, float) and (peakB.xCount - peakA.xCount) in [1, 2, 3] and peakB.loading == peakA.loading:
                                # different number of 1 atoms (peakA has less than peakB)
                                if a not in todel.keys():
                                    todel[a] = []
                                todel[a].append("same mz but reduced number of carbon atoms with " + str(peakB.mz) + " " + str(peakB.xCount))

                            if isinstance(peakA.xCount, float) and (peakA.xCount * 2) == peakB.xCount and peakA.loading == peakB.loading:
                                # a is intermediate pairing of polymer
                                if a not in todel.keys():
                                    todel[a] = []
                                todel[a].append("polymer mismatch with half the number of carbon atoms with " + str(peakB.mz) + " " + str(peakB.xCount))

                            if isinstance(peakA.xCount, float) and (peakA.xCount * 2) == peakB.xCount and peakA.loading == 1 and peakB.loading == 2:
                                # a is intermediate pairing of polymer
                                if a not in todel.keys():
                                    todel[a] = []
                                todel[a].append("singly charged mismatch with half the number of carbon atoms with " + str(peakB.mz) + " " + str(peakB.xCount))

                            if isinstance(peakA.xCount, float) and 0 <= (peakB.xCount - peakA.xCount * 3) <= 2 and peakA.loading == 1 and peakB.loading == 3:
                                # a is intermediate pairing of polymer
                                if a not in todel.keys():
                                    todel[a] = []
                                todel[a].append("singly charged mismatch with third the number of carbon atoms with " + str(peakB.mz) + " " + str(peakB.xCount))

                        elif abs(peakB.mz - peakA.mz - 1.00335 / peakA.loading) <= (peakA.mz * 2.5 * self.ppm / 1000000.0) and peakB.xCount == peakA.xCount and peakB.NPeakArea < 2 * peakA.NPeakArea:
                            ## increased m/z value by one carbon atom, but the same number of labeling atoms ## happens quite often with 15N-labeled metabolites e.g. 415.21374 and 415.7154 or 414.6978 and 415.19966
                            if b not in todel.keys():
                                todel[b] = []
                            todel[b].append("increased m/z by one labeling atoms while the number of labeled atoms is identical." + str(peakA.mz))

                        else:
                            for i in [1, 2, 3]:
                                if peakA.loading == peakB.loading and abs(peakB.mz - peakA.mz - i * self.xOffset / peakA.loading) <= peakA.mz * 2.5 * self.ppm / 1000000.0 and isinstance(peakA.xCount, float) and (peakA.xCount - peakB.xCount) in [1, 2, 3]:
                                    # b has an mz offset and a reduced number of carbon atoms (by one labelling atom)
                                    if b not in todel.keys():
                                        todel[b] = []
                                    todel[b].append("increased mz (by %d) and reduced number of labeling atoms (by %d) with " % (i, peakA.xCount - peakB.xCount) + str(peakA.mz) + " " + str(peakA.xCount))
                                elif peakA.loading == peakB.loading and abs(peakB.mz - peakA.mz - i * 1.00335 / peakA.loading) <= peakA.mz * 2.5 * self.ppm / 1000000.0 and peakA.xCount == peakB.xCount and peakB.NPeakArea <= peakA.NPeakArea * 0.1:
                                    # b has an mz offset and a reduced number of carbon atoms (by one labelling atom)
                                    if b not in todel.keys():
                                        todel[b] = []
                                    todel[b].append("increased mz (by %d 13C) and same number of labeling atoms with " % (i) + str(peakA.mz) + " " + str(peakA.xCount))

        for a in range(0, len(chromPeaks)):
            for b in range(0, len(chromPeaks)):
                peakA = chromPeaks[a]
                peakB = chromPeaks[b]
                if a != b and peakA.ionMode == peakB.ionMode:
                    if abs(peakA.NPeakCenter - peakB.NPeakCenter) <= self.peakCenterError:
                        # same chrom peak
                        if abs(peakA.mz - peakB.mz - (peakA.lmz - peakA.mz)) <= peakA.mz * 2 * self.ppm / 1000000.0 and ((isinstance(peakA.xCount, float) and abs(peakA.xCount * 2 - peakB.xCount) <= 1) or (peakA.xCount == peakB.xCount)):
                            if a not in todel.keys():
                                todel[a] = []
                            todel[a].append("dimer (1)" + str(peakB.mz) + " " + str(peakB.xCount))
        delet = []
        for x in todel.keys():
            delet.append(x)
        delet.sort()
        delet.reverse()
        for dele in delet:
            cp = chromPeaks.pop(dele)
            # Delete from chromPeaks
            db_con.tables["chromPeaks"] = db_con.tables["chromPeaks"].filter(pl.col("id") != cp.id)
            # Update allChromPeaks comment
            db_con.tables["allChromPeaks"] = db_con.tables["allChromPeaks"].with_columns(pl.when(pl.col("id") == cp.id).then(pl.lit(",".join(todel[dele]))).otherwise(pl.col("comment")).alias("comment"))

        # Delete XICs not in chromPeaks
        valid_eic_ids = db_con.tables["chromPeaks"]["eicID"].unique().to_list()
        db_con.tables["XICs"] = db_con.tables["XICs"].filter(pl.col("id").is_in(valid_eic_ids))

        db_con.commit()
        db_con.close()

    def _get_isotopolog_mass_diffs_from_formula_tools(self):
        elem_details = formulaTools().elemDetails

        required_isotopes = ["12C", "13C", "14N", "15N", "32S", "34S", "56Fe", "54Fe", "35Cl", "37Cl", "1H", "D", "16O", "18O"]
        missing = [iso for iso in required_isotopes if iso not in elem_details]
        if missing:
            raise RuntimeError("Missing isotope definitions in formulaTools dictionary: %s" % ", ".join(missing))

        return {
            "13C": elem_details["13C"][3] - elem_details["12C"][3],
            "15N": elem_details["15N"][3] - elem_details["14N"][3],
            "34S": elem_details["34S"][3] - elem_details["32S"][3],
            "54Fe": elem_details["54Fe"][3] - elem_details["56Fe"][3],
            "37Cl": elem_details["37Cl"][3] - elem_details["35Cl"][3],
            "D": elem_details["D"][3] - elem_details["1H"][3],
            "18O": elem_details["18O"][3] - elem_details["16O"][3],
        }

    def calculateIsotopologRatiosForFeaturePairs(self, chromPeaks, mzxml, reportFunction=None):
        """For each detected feature pair, extract EICs for isotopologs of M and M',
        calculate peak area ratios within the respective peak boundaries, and store
        the results as a dictionary in peak.isotopologRatios.

        All M-based ratios are relative to the M peak area; all M'-based ratios are
        relative to the M' peak area.
        """
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        def _peak_area(eic, times, lb, rb):
            rb = min(rb, len(eic) - 1)
            lb = max(lb, 0)
            if lb >= rb:
                return 0.0
            return float(np.trapezoid(eic[lb : rb + 1], times[lb : rb + 1]))

        mass_diffs = self._get_isotopolog_mass_diffs_from_formula_tools()
        m_isotopolog_offsets = [
            ("M-13C2", -2 * mass_diffs["13C"]),
            ("M-13C", -1 * mass_diffs["13C"]),
            ("M+13C", +1 * mass_diffs["13C"]),
            ("M+13C2", +2 * mass_diffs["13C"]),
            ("M+15N", +1 * mass_diffs["15N"]),
            ("M+34S", +1 * mass_diffs["34S"]),
            ("M-54Fe", +1 * mass_diffs["54Fe"]),
            ("M+37Cl", +1 * mass_diffs["37Cl"]),
        ]
        mp_isotopolog_offsets = [
            ("Mp-13C2", -2 * mass_diffs["13C"]),
            ("Mp-13C", -1 * mass_diffs["13C"]),
            ("Mp+13C", +1 * mass_diffs["13C"]),
            ("Mp+13C2", +2 * mass_diffs["13C"]),
            ("Mp+15N", +1 * mass_diffs["15N"]),
            ("Mp+34S", +1 * mass_diffs["34S"]),
            ("Mp-54Fe", +1 * mass_diffs["54Fe"]),
            ("Mp+37Cl", +1 * mass_diffs["37Cl"]),
            ("Mp+D", +1 * mass_diffs["D"]),
            ("Mp+18O", +1 * mass_diffs["18O"]),
        ]

        for i, peak in enumerate(chromPeaks):
            if reportFunction is not None:
                reportFunction(1.0 * i / len(chromPeaks), "%d features remaining" % (len(chromPeaks) - i))

            scanEvent = self.positiveScanEvent if peak.ionMode == "+" else self.negativeScanEvent
            loading = peak.loading

            lb_N = max(0, peak.NPeakCenter - int(peak.NBorderLeft))
            rb_N = peak.NPeakCenter + int(peak.NBorderRight)
            lb_L = max(0, peak.LPeakCenter - int(peak.LBorderLeft))
            rb_L = peak.LPeakCenter + int(peak.LBorderRight)

            # Extract reference EICs for M and M'
            eic_M, times_M, _, _ = mzxml.getEIC(peak.mz, self.chromPeakPPM, filterLine=scanEvent)
            eic_Mp, times_Mp, _, _ = mzxml.getEIC(peak.lmz, self.chromPeakPPM, filterLine=scanEvent)
            eic_M = np.asarray(eic_M, dtype=np.float64)
            eic_Mp = np.asarray(eic_Mp, dtype=np.float64)
            times_M = np.asarray(times_M, dtype=np.float64)
            times_Mp = np.asarray(times_Mp, dtype=np.float64)

            area_M = _peak_area(eic_M, times_M, lb_N, rb_N)
            area_Mp = _peak_area(eic_Mp, times_Mp, lb_L, rb_L)

            isotopolog_ratios = {}

            for label, offset in m_isotopolog_offsets:
                target_mz = peak.mz + offset / loading
                eic_iso, times_iso, _, _ = mzxml.getEIC(target_mz, self.chromPeakPPM, filterLine=scanEvent)
                eic_iso = np.asarray(eic_iso, dtype=np.float64)
                times_iso = np.asarray(times_iso, dtype=np.float64)
                area_iso = _peak_area(eic_iso, times_iso, lb_N, rb_N)
                isotopolog_ratios[label] = area_iso / area_M if area_M > 0 else 0.0

            for label, offset in mp_isotopolog_offsets:
                target_mz = peak.lmz + offset / loading
                eic_iso, times_iso, _, _ = mzxml.getEIC(target_mz, self.chromPeakPPM, filterLine=scanEvent)
                eic_iso = np.asarray(eic_iso, dtype=np.float64)
                times_iso = np.asarray(times_iso, dtype=np.float64)
                area_iso = _peak_area(eic_iso, times_iso, lb_L, rb_L)
                isotopolog_ratios[label] = area_iso / area_Mp if area_Mp > 0 else 0.0

            peak.isotopologRatios = isotopolog_ratios

        # Persist to DB
        for peak in chromPeaks:
            encoded = base64.b64encode(dumps(peak.isotopologRatios)).decode("utf-8")
            db_con.tables["chromPeaks"] = db_con.tables["chromPeaks"].with_columns(pl.when(pl.col("id") == peak.id).then(pl.lit(encoded)).otherwise(pl.col("isotopologRatios")).alias("isotopologRatios"))
            db_con.tables["allChromPeaks"] = db_con.tables["allChromPeaks"].with_columns(pl.when(pl.col("id") == peak.id).then(pl.lit(encoded)).otherwise(pl.col("isotopologRatios")).alias("isotopologRatios"))

        self.printMessage("Isotopolog ratio calculation done.", type="info")
        db_con.commit()
        db_con.close()

    def calculateIsotopologEnrichmentForFeaturePairs(self, chromPeaks, mzxml, reportFunction=None):
        """For each detected feature pair, quantify the individual isotopologs from M-2 to M'+2
        (named "M+x / M'-(Xn-x)", with x the number of labelling atoms), determine how many of
        them are present (area >= 1% of the less abundant of M/M'), and calculate the isotopic
        enrichment from the native (M and the highest detected M+x) and labelled (M' and M'-1)
        isotopolog pairs.

        Results are stored on the peak as isoAreas (dict of "M+x / M'-(Xn-x)" -> area),
        isoEnrichment (dict with keys "M" and "M'") and isoCount (number of isotopologs detected
        between M and M').
        """
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        def _peak_area(eic, times, lb, rb):
            rb = min(rb, len(eic) - 1)
            lb = max(lb, 0)
            if lb >= rb:
                return 0.0
            return float(np.trapezoid(eic[lb : rb + 1], times[lb : rb + 1]))

        for i, peak in enumerate(chromPeaks):
            if reportFunction is not None:
                reportFunction(1.0 * i / len(chromPeaks), "%d features remaining" % (len(chromPeaks) - i))

            peak.isoAreas = {}
            peak.isoEnrichment = {}
            peak.isoCount = 0

            try:
                xCount = int(peak.xCount)
            except (TypeError, ValueError):
                continue
            if xCount < 1:
                continue

            scanEvent = self.positiveScanEvent if peak.ionMode == "+" else self.negativeScanEvent
            loading = peak.loading

            # common integration border spanning both the M and M' peak boundaries
            lb = max(0, min(peak.NPeakCenter - int(peak.NBorderLeft), peak.LPeakCenter - int(peak.LBorderLeft)))
            rb = max(peak.NPeakCenter + int(peak.NBorderRight), peak.LPeakCenter + int(peak.LBorderRight))

            eic_M, times_M, _, _ = mzxml.getEIC(peak.mz, self.chromPeakPPM, filterLine=scanEvent)
            eic_Mp, times_Mp, _, _ = mzxml.getEIC(peak.lmz, self.chromPeakPPM, filterLine=scanEvent)
            eic_M = np.asarray(eic_M, dtype=np.float64)
            eic_Mp = np.asarray(eic_Mp, dtype=np.float64)
            times_M = np.asarray(times_M, dtype=np.float64)
            times_Mp = np.asarray(times_Mp, dtype=np.float64)

            area_M = _peak_area(eic_M, times_M, lb, rb)
            area_Mp = _peak_area(eic_Mp, times_Mp, lb, rb)

            if area_M <= 0 and area_Mp <= 0:
                continue

            refArea = min(area_M, area_Mp) if area_M > 0 and area_Mp > 0 else max(area_M, area_Mp)
            threshold = 0.01 * refArea

            areasByX = {}
            for x in range(-2, xCount + 3):
                if x == 0:
                    areasByX[x] = area_M
                elif x == xCount:
                    areasByX[x] = area_Mp
                else:
                    target_mz = peak.mz + x * self.xOffset / loading
                    eic_x, times_x, _, _ = mzxml.getEIC(target_mz, self.chromPeakPPM, filterLine=scanEvent)
                    eic_x = np.asarray(eic_x, dtype=np.float64)
                    times_x = np.asarray(times_x, dtype=np.float64)
                    areasByX[x] = _peak_area(eic_x, times_x, lb, rb)

            isoAreas = {isotopologLabel(x, xCount): a for x, a in areasByX.items() if a >= threshold}
            peak.isoCount = sum(1 for x in range(1, xCount) if areasByX.get(x, 0.0) >= threshold)
            peak.isoAreas = isoAreas

            # native-side enrichment: use M and the highest contiguously-detected M+x
            nativeX = 0
            for x in range(1, xCount):
                if areasByX.get(x, 0.0) >= threshold:
                    nativeX = x
                else:
                    break

            isoEnrichment = {}
            if nativeX >= 1 and area_M > 0:
                isoEnrichment["M"] = calcIsoEnrichment(xCount, nativeX, areasByX[nativeX] / area_M)

            # labelled-side enrichment: use M' and M'-1 (M itself if xCount == 1)
            areaMpMinus1 = areasByX.get(xCount - 1, area_M) if xCount >= 2 else area_M
            if area_Mp > 0 and areaMpMinus1 > 0:
                isoEnrichment["M'"] = calcIsoEnrichment(xCount, 1, areaMpMinus1 / area_Mp)

            peak.isoEnrichment = isoEnrichment

        # Persist to DB
        for peak in chromPeaks:
            encodedAreas = base64.b64encode(dumps(getattr(peak, "isoAreas", {}))).decode("utf-8")
            encodedEnrichment = base64.b64encode(dumps(getattr(peak, "isoEnrichment", {}))).decode("utf-8")
            isoCount = getattr(peak, "isoCount", 0)
            for table in ("chromPeaks", "allChromPeaks"):
                db_con.tables[table] = db_con.tables[table].with_columns(
                    pl.when(pl.col("id") == peak.id).then(pl.lit(encodedAreas)).otherwise(pl.col("isoAreas")).alias("isoAreas"),
                    pl.when(pl.col("id") == peak.id).then(pl.lit(encodedEnrichment)).otherwise(pl.col("isoEnrichment")).alias("isoEnrichment"),
                    pl.when(pl.col("id") == peak.id).then(pl.lit(isoCount)).otherwise(pl.col("isoCount")).alias("isoCount"),
                )

        self.printMessage("Isotopolog enrichment calculation done.", type="info")
        db_con.commit()
        db_con.close()

    # data processing step 5: in full metabolome labeling experiments hetero atoms (e.g. S, Cl) may
    # show characterisitc isotope patterns on the labeled metabolite ion side. There, these peaks may not be
    # dominated by the usually much more abundant carbon isotopes and can thus be easier seen. However, for
    # low abundant metabolite ions these isotope peaks may not be present at all. The search is performed
    # on a MS scan level and does not directly use the chromatographic information (no chromatographic
    # peak picking is performed)
    def annotateFeaturePairs(self, chromPeaks, mzxml, tracer, reportFunction=None):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        self.postMessageToProgressWrapper("text", "%s: Annotating feature pairs" % tracer.name)
        for i in range(0, len(chromPeaks)):
            if reportFunction is not None:
                reportFunction(
                    1.0 * i / len(chromPeaks),
                    "%d features remaining" % (len(chromPeaks) - i),
                )

            peak = chromPeaks[i]

            ## Annotate hetero atoms
            scanEvent = ""
            if peak.ionMode == "+":
                scanEvent = self.positiveScanEvent
            elif peak.ionMode == "-":
                scanEvent = self.negativeScanEvent

            for pIso in self.heteroAtoms:
                pIsotope = self.heteroAtoms[pIso]

                mz = 0
                if pIsotope.mzOffset < 0:
                    mz = peak.mz + pIsotope.mzOffset / peak.loading  # delta m/z is negative, therefore this decreases the search m/z
                else:
                    mz = peak.lmz + pIsotope.mzOffset / peak.loading  # delta m/z is positive

                refMz = 0
                if pIsotope.mzOffset < 0:
                    refMz = peak.mz
                else:
                    refMz = peak.lmz

                for haCount in range(pIsotope.minCount, pIsotope.maxCount + 1):
                    if haCount == 0:
                        continue

                    for curScanNum in range(
                        int(
                            max(
                                peak.NPeakCenter - peak.NBorderLeft,
                                peak.LPeakCenter - peak.LBorderLeft,
                            )
                        ),
                        int(
                            min(
                                peak.NPeakCenter + peak.NBorderRight,
                                peak.LPeakCenter + peak.LBorderRight,
                            )
                        )
                        + 1,
                    ):
                        scan = mzxml.getIthMS1Scan(curScanNum, scanEvent)
                        if scan is not None:
                            mzBounds = scan.findMZ(refMz, self.ppm)
                            mzBounds = scan.getMostIntensePeak(mzBounds[0], mzBounds[1])
                            if mzBounds != -1:
                                peakIntensity = scan.intensity_list[mzBounds]

                                isoBounds = scan.findMZ(mz, self.ppm)
                                isoBounds = scan.getMostIntensePeak(isoBounds[0], isoBounds[1])
                                if isoBounds != -1:
                                    isoIntensity = scan.intensity_list[isoBounds]

                                    relativeIntensity = isoIntensity / peakIntensity
                                    if abs(relativeIntensity - pIsotope.relativeAbundance * haCount) < self.hAIntensityError:
                                        if pIso not in peak.heteroIsotopologues:
                                            peak.heteroIsotopologues[pIso] = {}
                                        if haCount not in peak.heteroIsotopologues[pIso]:
                                            peak.heteroIsotopologues[pIso][haCount] = []
                                        peak.heteroIsotopologues[pIso][haCount].append(
                                            (
                                                scan.id,
                                                relativeIntensity,
                                                pIsotope.relativeAbundance * haCount,
                                            )
                                        )

            rmHIso = []
            for hI in peak.heteroIsotopologues:
                for haCount in peak.heteroIsotopologues[hI]:
                    if len(peak.heteroIsotopologues[hI][haCount]) < self.hAMinScans:
                        rmHIso.append((hI, haCount))

            for rmHI, rmHACount in rmHIso:
                del peak.heteroIsotopologues[rmHI][rmHACount]
            rmHIso = []
            for hi in peak.heteroIsotopologues:
                if len(peak.heteroIsotopologues[hi]) == 0:
                    rmHIso.append(hi)
            for rmHI in rmHIso:
                del peak.heteroIsotopologues[rmHI]

            self.getMostLikelyHeteroIsotope(peak.heteroIsotopologues)

            ## Annotate isotopolog ratios

            def findRatiosForMZs(mzFrom, mzTo, fromScan, toScan, mzxml, scanEvent, ppm):
                isoRatios = []
                for curScanNum in range(fromScan, toScan):
                    scan = mzxml.getIthMS1Scan(curScanNum, scanEvent)
                    if scan is not None:
                        mzBounds = scan.findMZ(mzTo, ppm)
                        mzBounds = scan.getMostIntensePeak(mzBounds[0], mzBounds[1])
                        if mzBounds != -1:
                            peakIntensity = scan.intensity_list[mzBounds]

                            isoBounds = scan.findMZ(mzFrom, ppm)
                            isoBounds = scan.getMostIntensePeak(isoBounds[0], isoBounds[1])

                            if isoBounds != -1:
                                isoPeakIntensity = scan.intensity_list[isoBounds]

                                isoRatios.append(
                                    Bunch(
                                        ratio=isoPeakIntensity / peakIntensity,
                                        refInt=peakIntensity,
                                    )
                                )

                return isoRatios

            peak.isotopeRatios = []
            for i in range(1, self.calcIsoRatioNative + 1):
                isoRatios = findRatiosForMZs(
                    peak.mz + 1.00335484 * i / peak.loading,
                    peak.mz,
                    int(
                        max(
                            peak.NPeakCenter - peak.NBorderLeft,
                            peak.LPeakCenter - peak.LBorderLeft,
                        )
                    ),
                    int(
                        min(
                            peak.NPeakCenter + peak.NBorderRight,
                            peak.LPeakCenter + peak.LBorderRight,
                        )
                    )
                    + 1,
                    mzxml,
                    scanEvent,
                    self.ppm,
                )
                observedRatioMean = weightedMean([t.ratio for t in isoRatios], [t.refInt for t in isoRatios])
                observedRatioSD = weightedSd([t.ratio for t in isoRatios], [t.refInt for t in isoRatios])
                if isinstance(peak.xCount, str):
                    theoreticalRatio = -1
                else:
                    theoreticalRatio = getNormRatio(self.purityN, peak.xCount, i)
                peak.isotopeRatios.append(
                    Bunch(
                        type="native",
                        subs=i,
                        observedRatioMean=observedRatioMean,
                        observedRatioSD=observedRatioSD,
                        theoreticalRatio=theoreticalRatio,
                    )
                )

            for i in range(-1, self.calcIsoRatioLabelled - 1, -1):
                isoRatios = findRatiosForMZs(
                    peak.lmz + 1.00335484 * i / peak.loading,
                    peak.lmz,
                    int(
                        max(
                            peak.NPeakCenter - peak.NBorderLeft,
                            peak.LPeakCenter - peak.LBorderLeft,
                        )
                    ),
                    int(
                        min(
                            peak.NPeakCenter + peak.NBorderRight,
                            peak.LPeakCenter + peak.LBorderRight,
                        )
                    )
                    + 1,
                    mzxml,
                    scanEvent,
                    self.ppm,
                )
                observedRatioMean = weightedMean([t.ratio for t in isoRatios], [t.refInt for t in isoRatios])
                observedRatioSD = weightedSd([t.ratio for t in isoRatios], [t.refInt for t in isoRatios])
                if isinstance(peak.xCount, str):
                    theoreticalRatio = -1
                else:
                    theoreticalRatio = getNormRatio(self.purityL, peak.xCount, abs(i))
                peak.isotopeRatios.append(
                    Bunch(
                        type="labelled",
                        subs=abs(i),
                        observedRatioMean=observedRatioMean,
                        observedRatioSD=observedRatioSD,
                        theoreticalRatio=theoreticalRatio,
                    )
                )
            if self.metabolisationExperiment:
                for i in range(1, self.calcIsoRatioMoiety + 1):
                    isoRatios = findRatiosForMZs(
                        peak.lmz + i * 1.00335484 / peak.loading,
                        peak.lmz,
                        int(
                            max(
                                peak.NPeakCenter - peak.NBorderLeft,
                                peak.LPeakCenter - peak.LBorderLeft,
                            )
                        ),
                        int(
                            min(
                                peak.NPeakCenter + peak.NBorderRight,
                                peak.LPeakCenter + peak.LBorderRight,
                            )
                        )
                        + 1,
                        mzxml,
                        scanEvent,
                        self.ppm,
                    )
                    observedRatioMean = weightedMean([t.ratio for t in isoRatios], [t.refInt for t in isoRatios])
                    observedRatioSD = weightedSd([t.ratio for t in isoRatios], [t.refInt for t in isoRatios])
                    peak.isotopeRatios.append(
                        Bunch(
                            type="moiety",
                            subs=i,
                            observedRatioMean=observedRatioMean,
                            observedRatioSD=observedRatioSD,
                            theoreticalRatio=0,
                        )
                    )

            def findMZDifferenceRelativeToXnForMZs(mzFrom, mzTo, tmz, fromScan, toScan, mzxml, scanEvent, ppm):
                diffs = []
                for curScanNum in range(fromScan, toScan):
                    scan = mzxml.getIthMS1Scan(curScanNum, scanEvent)
                    if scan is not None:
                        mzBounds = scan.findMZ(mzTo, ppm)
                        mzBounds = scan.getMostIntensePeak(mzBounds[0], mzBounds[1])
                        if mzBounds != -1:
                            peakMZ = scan.mz_list[mzBounds]

                            refBounds = scan.findMZ(mzFrom, ppm)
                            refBounds = scan.getMostIntensePeak(refBounds[0], refBounds[1])

                            if refBounds != -1:
                                isoPeakMZ = scan.mz_list[refBounds]

                                diffs.append((abs(isoPeakMZ - peakMZ) - tmz) * 1000000.0 / mzFrom)

                return diffs

            diffs = findMZDifferenceRelativeToXnForMZs(
                peak.mz,
                peak.lmz,
                peak.lmz - peak.mz,
                int(
                    max(
                        peak.NPeakCenter - peak.NBorderLeft,
                        peak.LPeakCenter - peak.LBorderLeft,
                    )
                ),
                int(
                    min(
                        peak.NPeakCenter + peak.NBorderRight,
                        peak.LPeakCenter + peak.LBorderRight,
                    )
                )
                + 1,
                mzxml,
                scanEvent,
                self.ppm,
            )
            peak.mzDiffErrors = Bunch(mean=mean(diffs), sd=sd(diffs), vals=diffs)

        # Batch update using Polars for better performance
        updates_dict = {"heteroAtoms": {}, "isotopesRatios": {}, "mzDiffErrors": {}}

        for i in range(len(chromPeaks)):
            peak = chromPeaks[i]
            updates_dict["heteroAtoms"][peak.id] = base64.b64encode(dumps(peak.heteroIsotopologues)).decode("utf-8")
            updates_dict["isotopesRatios"][peak.id] = base64.b64encode(dumps(peak.isotopeRatios)).decode("utf-8")
            updates_dict["mzDiffErrors"][peak.id] = base64.b64encode(dumps(peak.mzDiffErrors)).decode("utf-8")

        # Update chromPeaks
        for col_name, id_val_map in updates_dict.items():
            for peak_id, value in id_val_map.items():
                db_con.tables["chromPeaks"] = db_con.tables["chromPeaks"].with_columns(pl.when(pl.col("id") == peak_id).then(pl.lit(value)).otherwise(pl.col(col_name)).alias(col_name))
                db_con.tables["allChromPeaks"] = db_con.tables["allChromPeaks"].with_columns(pl.when(pl.col("id") == peak_id).then(pl.lit(value)).otherwise(pl.col(col_name)).alias(col_name))

        self.printMessage("Annotating feature pairs done.", type="info")

        db_con.commit()
        db_con.close()

    # annotate a metabolite group (consisting of ChromPeakPair instances) with the defined
    # hetero atoms based on detected feature pairs
    def annotateFeaturePairsWithHeteroAtoms(self, group, peaksInGroup):
        # iterate all peaks pairwise to find different adducts of the metabolite
        for pa in group:
            peakA = peaksInGroup[pa]

            for pb in group:
                peakB = peaksInGroup[pb]

                # peakA shall always have the lower mass
                if peakA.mz < peakB.mz and peakA.xCount == peakB.xCount:
                    ## check, if it could be a hetero atom
                    for pIso in self.heteroAtoms:
                        pIsotope = self.heteroAtoms[pIso]

                        bestFit = None
                        bestFitRatio = 100000
                        bestFitID = None

                        if abs(peakB.mz - peakA.mz - pIsotope.mzOffset) <= (max(peakA.mz, peakB.mz) * 2.5 * self.ppm / 1000000.0):
                            if pIsotope.mzOffset > 0:
                                ratio = peakB.LPeakArea / peakA.LPeakArea
                            else:
                                ratio = peakB.NPeakArea / peakA.NPeakArea

                            for nHAtoms in range(pIsotope.minCount, pIsotope.maxCount + 1):
                                if abs(ratio - nHAtoms * pIsotope.relativeAbundance) <= self.hAIntensityError:
                                    if abs(ratio - nHAtoms * pIsotope.relativeAbundance) < bestFitRatio:
                                        bestFitRatio = abs(ratio - nHAtoms * pIsotope.relativeAbundance)
                                        bestFit = Bunch(pIso=pIso, nHAtoms=nHAtoms)
                                        bestFitID = pb

                        if bestFit is not None:
                            peakB = peaksInGroup[bestFitID]
                            # peakA.heteroAtomsFeaturePairs.append(Bunch(name="M_%s%d"%(pIso, bestFit),  partnerAdd="_%s%d"%(pIso, bestFit),  toPeak=pb))
                            peakB.heteroAtomsFeaturePairs.append(
                                Bunch(
                                    name="_%s%d" % (bestFit.pIso, bestFit.nHAtoms),
                                    partnerAdd="M_%s%d" % (bestFit.pIso, bestFit.nHAtoms),
                                    toPeak=pa,
                                )
                            )

    # annotate a metabolite group (consisting of ChromPeakPair instances) with the defined
    # adducts and generate possible in-source fragments. Remove inconsistencies
    # in the form of impossible adduct combinations (e.g. [M+H]+ and [M+Na]+ for the same ion)
    def annotateChromPeaks(self, group, peaksInGroup):
        for pe in group:
            peak = peaksInGroup[pe]

            if not hasattr(peak, "fDesc"):
                setattr(peak, "fDesc", [])
            if not hasattr(peak, "adducts"):
                setattr(peak, "adducts", [])
            if not hasattr(peak, "Ms"):
                setattr(peak, "Ms", [])

        if len(group) <= 40 or True:
            self.annotateFeaturePairsWithHeteroAtoms(group, peaksInGroup)
            for pa in group:
                peakA = peaksInGroup[pa]
                peakA.adducts.extend(peakA.heteroAtomsFeaturePairs)

            fT = formulaTools()

            # prepare adducts list
            addPol = {}
            addPol["+"] = []
            addPol["-"] = []
            adductsDict = {}
            for adduct in self.adducts:
                adductsDict[adduct.name] = adduct
                if adduct.polarity != "":
                    addPol[adduct.polarity].append(adduct)
            adducts = addPol
            adducts["+"] = sorted(adducts["+"], key=lambda x: x.mzoffset)
            adducts["-"] = sorted(adducts["-"], key=lambda x: x.mzoffset)

            # 1. iterate all peaks pairwise to find different adducts of the metabolite
            for pa in group:
                peakA = peaksInGroup[pa]
                for pb in group:
                    peakB = peaksInGroup[pb]

                    # peakA shall always have the lower mass
                    if peakA.mz < peakB.mz and (len(peakB.heteroAtomsFeaturePairs) == 0 or not all([s.name.startswith("_") for s in peakB.heteroAtomsFeaturePairs])):
                        # search for different adduct combinations
                        # different adducts must have the same number of labelled atoms
                        if peakA.xCount == peakB.xCount:
                            # check, if it could be some kind of adduct combination
                            for adA in adducts[peakA.ionMode]:
                                for adB in adducts[peakB.ionMode]:
                                    if adA.mCount == 1 and adB.mCount == 1 and adA.mzoffset < adB.mzoffset:
                                        if (
                                            peakA.ionMode == "-"
                                            and peakB.ionMode == "+"
                                            and peakA.loading == peakB.loading
                                            and abs(abs(adB.mzoffset - adA.mzoffset) - 2 * 1.007276) <= (max(peakA.mz, peakB.mz) * 2.5 * self.ppm / 1000000.0)
                                            and abs(peakB.mz - peakA.mz - 2 * 1.007276 / peakA.loading) <= (max(peakA.mz, peakB.mz) * 2.5 * self.ppm / 1000000.0)
                                        ):
                                            peakA.adducts.append(
                                                Bunch(
                                                    name="[M-H]-",
                                                    partnerAdd="[M+H]+",
                                                    toPeak=pb,
                                                )
                                            )
                                            peakB.adducts.append(
                                                Bunch(
                                                    name="[M+H]+",
                                                    partnerAdd="[M-H]-",
                                                    toPeak=pa,
                                                )
                                            )

                                        elif peakA.loading == adA.charge and peakB.loading == adB.charge and abs((peakA.mz - adA.mzoffset) / adA.mCount * peakA.loading - (peakB.mz - adB.mzoffset) / adB.mCount * peakB.loading) <= (max(peakA.mz, peakB.mz) * 2.5 * self.ppm / 1000000.0):
                                            peakA.adducts.append(
                                                Bunch(
                                                    name=adA.name,
                                                    partnerAdd=adB.name,
                                                    toPeak=pb,
                                                )
                                            )
                                            peakB.adducts.append(
                                                Bunch(
                                                    name=adB.name,
                                                    partnerAdd=adA.name,
                                                    toPeak=pa,
                                                )
                                            )

                        # search for adducts of the form [2M+XX]+-
                        elif peakA.xCount * 2 == peakB.xCount:
                            for adA in adducts[peakA.ionMode]:
                                for adB in adducts[peakB.ionMode]:
                                    if adA.mCount == 1 and adB.mCount == 2:
                                        if peakA.loading == adA.charge and peakB.loading == adB.charge and abs((peakA.mz - adA.mzoffset) / adA.mCount * peakA.loading - (peakB.mz - adB.mzoffset) / adB.mCount * peakB.loading) <= (max(peakA.mz, peakB.mz) * 2.5 * self.ppm / 1000000.0):
                                            peakA.adducts.append(
                                                Bunch(
                                                    name=adA.name,
                                                    partnerAdd=adB.name,
                                                    toPeak=pb,
                                                )
                                            )
                                            peakB.adducts.append(
                                                Bunch(
                                                    name=adB.name,
                                                    partnerAdd=adA.name,
                                                    toPeak=pa,
                                                )
                                            )

            def isAdductPairPresent(pA, adductAName, adductsA, pB, adductBName, adductsB, checkPartners=True):
                aFound = False
                for add in adductsA:
                    if add.name == adductAName and (not checkPartners or (add.partnerAdd == adductBName and add.toPeak == pB)):
                        aFound = True
                bFound = False
                for add in adductsB:
                    if add.name == adductBName and (not checkPartners or (add.partnerAdd == adductAName and add.toPeak == pA)):
                        bFound = True
                return aFound and bFound

            def removeAdductFromFeaturePair(pA, adductAName, adductsA):
                aFound = []
                for i, add in enumerate(adductsA):
                    if add.name == adductAName:
                        aFound.append(i)

                aFound = reversed(sorted(list(set(aFound))))
                for i in aFound:
                    del adductsA[i]

            def removeAdductPair(pA, adductAName, adductsA, pB, adductBName, adductsB):
                aFound = []
                for i, add in enumerate(adductsA):
                    if add.name == adductAName and add.partnerAdd == adductBName and add.toPeak == pB:
                        aFound.append(i)
                bFound = []
                for i, add in enumerate(adductsB):
                    if add.name == adductBName and add.partnerAdd == adductAName and add.toPeak == pA:
                        bFound.append(i)

                aFound = reversed(sorted(list(set(aFound))))
                for i in aFound:
                    del adductsA[i]
                bFound = reversed(sorted(list(set(bFound))))
                for i in bFound:
                    del adductsB[i]

            # 2. remove incorrect annotations from feature pairs e.g. A: [M+H]+ and [M+Na]+ with B: [M+Na]+ and [M+2Na-H]+
            for p in group:
                peak = peaksInGroup[p]
                peak.adducts = list(set(peak.adducts))

            for pa in group:
                peakA = peaksInGroup[pa]
                if len(peakA.adducts) > 0:
                    for pb in group:
                        peakB = peaksInGroup[pb]
                        if len(peakB.adducts) > 0:
                            if peakA.mz < peakB.mz:
                                for pc in group:
                                    peakC = peaksInGroup[pc]
                                    if len(peakC.adducts) > 0:
                                        if (
                                            isAdductPairPresent(
                                                pa,
                                                "[M+H]+",
                                                peakA.adducts,
                                                pb,
                                                "[M+Na]+",
                                                peakB.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pa,
                                                "[M+Na]+",
                                                peakA.adducts,
                                                pb,
                                                "[M+2Na-H]+",
                                                peakB.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pa,
                                                "[M+H]+",
                                                peakA.adducts,
                                                pc,
                                                "[M+2Na-H]+",
                                                peakC.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pb,
                                                "[M+H]+",
                                                peakB.adducts,
                                                pc,
                                                "[M+Na]+",
                                                peakC.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pb,
                                                "[M+Na]+",
                                                peakB.adducts,
                                                pc,
                                                "[M+2Na-H]+",
                                                peakC.adducts,
                                            )
                                            and peakA.mz < peakB.mz < peakC.mz
                                        ):
                                            removeAdductPair(
                                                pa,
                                                "[M+Na]+",
                                                peakA.adducts,
                                                pb,
                                                "[M+2Na-H]+",
                                                peakB.adducts,
                                            )
                                            removeAdductPair(
                                                pb,
                                                "[M+H]+",
                                                peakB.adducts,
                                                pc,
                                                "[M+Na]+",
                                                peakC.adducts,
                                            )

                                        if (
                                            isAdductPairPresent(
                                                pa,
                                                "[M+H]+",
                                                peakA.adducts,
                                                pb,
                                                "[M+K]+",
                                                peakB.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pa,
                                                "[M+K]+",
                                                peakA.adducts,
                                                pb,
                                                "[M+2K-H]+",
                                                peakB.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pa,
                                                "[M+H]+",
                                                peakA.adducts,
                                                pc,
                                                "[M+2K-H]+",
                                                peakC.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pb,
                                                "[M+H]+",
                                                peakB.adducts,
                                                pc,
                                                "[M+K]+",
                                                peakC.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pb,
                                                "[M+K]+",
                                                peakB.adducts,
                                                pc,
                                                "[M+2K-H]+",
                                                peakC.adducts,
                                            )
                                            and peakA.mz < peakB.mz < peakC.mz
                                        ):
                                            removeAdductPair(
                                                pa,
                                                "[M+K]+",
                                                peakA.adducts,
                                                pb,
                                                "[M+2K-H]+",
                                                peakB.adducts,
                                            )
                                            removeAdductPair(
                                                pb,
                                                "[M+H]+",
                                                peakB.adducts,
                                                pc,
                                                "[M+K]+",
                                                peakC.adducts,
                                            )

                                        if (
                                            isAdductPairPresent(
                                                pa,
                                                "[M+2H]++",
                                                peakA.adducts,
                                                pb,
                                                "[M+H+Na]++",
                                                peakB.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pa,
                                                "[M+H+Na]++",
                                                peakA.adducts,
                                                pb,
                                                "[M+2Na]++",
                                                peakB.adducts,
                                            )
                                            and peakA.mz < peakB.mz
                                        ):
                                            removeAdductPair(
                                                pa,
                                                "[M+H+Na]++",
                                                peakA.adducts,
                                                pb,
                                                "[M+2Na]++",
                                                peakB.adducts,
                                            )

                                        if (
                                            isAdductPairPresent(
                                                pa,
                                                "[M+2H]++",
                                                peakA.adducts,
                                                pb,
                                                "[M+H+K]++",
                                                peakB.adducts,
                                            )
                                            and isAdductPairPresent(
                                                pa,
                                                "[M+H+K]++",
                                                peakA.adducts,
                                                pb,
                                                "[M+2K]++",
                                                peakB.adducts,
                                            )
                                            and peakA.mz < peakB.mz
                                        ):
                                            removeAdductPair(
                                                pa,
                                                "[M+H+Na]++",
                                                peakA.adducts,
                                                pb,
                                                "[M+2Na]++",
                                                peakB.adducts,
                                            )

            for pa in group:
                peakA = peaksInGroup[pa]
                if len(peakA.adducts) > 0:
                    if isAdductPairPresent(
                        pa,
                        "[M+H]+",
                        peakA.adducts,
                        pa,
                        "[2M+H]+",
                        peakA.adducts,
                        checkPartners=False,
                    ):
                        removeAdductFromFeaturePair(pa, "[M+H]+", peakA.adducts)
                    if isAdductPairPresent(
                        pa,
                        "[M+NH4]+",
                        peakA.adducts,
                        pa,
                        "[2M+NH4]+",
                        peakA.adducts,
                        checkPartners=False,
                    ):
                        removeAdductFromFeaturePair(pa, "[M+NH4]+", peakA.adducts)
                    if isAdductPairPresent(
                        pa,
                        "[M+Na]+",
                        peakA.adducts,
                        pa,
                        "[2M+Na]+",
                        peakA.adducts,
                        checkPartners=False,
                    ):
                        removeAdductFromFeaturePair(pa, "[M+Na]+", peakA.adducts)
                    if isAdductPairPresent(
                        pa,
                        "[M+K]+",
                        peakA.adducts,
                        pa,
                        "[2M+K]+",
                        peakA.adducts,
                        checkPartners=False,
                    ):
                        removeAdductFromFeaturePair(pa, "[M+K]+", peakA.adducts)

            inSourceFragments = {}

            # 3. iterate all peaks pairwise to find common in-source fragments
            for pa in group:
                peakA = peaksInGroup[pa]

                for pb in group:
                    peakB = peaksInGroup[pb]

                    # peakA shall always have the lower mass
                    if peakA.mz < peakB.mz and abs(peakA.mz - peakB.mz) <= 101:
                        mzDif = peakB.mz - peakA.mz

                        # generate putative in-source fragments (using the labelled carbon atoms)
                        if len(self.elements) > 0:
                            elemDictReq = {}
                            for elem in self.elements:
                                elemDictReq[elem] = [
                                    self.elements[elem].weight,
                                    self.elements[elem].numberValenzElectrons,
                                ]
                            t = SGRGenerator(atoms=elemDictReq)

                            useAtoms = []
                            if self.labellingElement in self.elements.keys():
                                useAtoms.append(self.labellingElement)
                                for k in self.elements.keys():
                                    if k != self.labellingElement:
                                        useAtoms.append(k)
                            else:
                                useAtoms = list(self.elements.keys())

                            try:
                                atomsRange = []
                                if self.labellingElement in useAtoms:
                                    if self.metabolisationExperiment:
                                        elem = self.elements[self.labellingElement]
                                        atomsRange.append(
                                            [
                                                abs(peakA.xCount - peakB.xCount),
                                                abs(peakA.xCount - peakB.xCount + elem.maxCount - elem.minCount),
                                            ]
                                        )
                                    else:
                                        atomsRange.append(abs(peakA.xCount - peakB.xCount))

                                for elem in self.elements.keys():
                                    if elem != self.labellingElement:
                                        elem = self.elements[elem]
                                        atomsRange.append([elem.minCount, elem.maxCount])

                                corrFact = abs(peakB.mz - peakA.mz)
                                if corrFact <= 1:
                                    corrFact = 1.0

                                pFs = t.findFormulas(
                                    mzDif,
                                    ppm=self.ppm * 2.0 * peakA.mz / corrFact,
                                    useAtoms=useAtoms,
                                    atomsRange=atomsRange,
                                    fixed=self.labellingElement,
                                    useSevenGoldenRules=False,
                                )

                                for pF in pFs:
                                    if pa not in inSourceFragments.keys():
                                        inSourceFragments[pa] = {}
                                    if pb not in inSourceFragments[pa].keys():
                                        inSourceFragments[pa][pb] = []

                                    c = fT.parseFormula(pF)
                                    mw = fT.calcMolWeight(c)
                                    abs(abs(peakB.mz - peakA.mz) - mw)
                                    sf = fT.flatToString(c, prettyPrintWithHTMLTags=False)
                                    inSourceFragments[pa][pb].append("%.4f-%s" % (peakB.mz, sf))

                            except Exception as ex:
                                print(f"ERROR: generating in-source fragments failed: {ex}")

            if self.simplifyInSourceFragments:
                for pa in group:
                    peakA = peaksInGroup[pa]
                    peakA.fDesc = []

                    for pb in group:
                        peakB = peaksInGroup[pb]

                        for pc in group:
                            peakC = peaksInGroup[pc]

                            if peakA.mz < peakC.mz < peakB.mz:
                                if pa in inSourceFragments.keys() and pb in inSourceFragments[pa].keys() and pc in inSourceFragments[pa].keys() and pc in inSourceFragments.keys() and pb in inSourceFragments[pc].keys():
                                    del inSourceFragments[pa][pc]

            for pa in group:
                peakA = peaksInGroup[pa]

                if pa in inSourceFragments.keys():
                    for pb in inSourceFragments[pa].keys():
                        for inFrag in inSourceFragments[pa][pb]:
                            peakA.fDesc.append(inFrag)

            for pe in group:
                peak = peaksInGroup[pe]
                peak.fDesc = list(set(peak.fDesc))
                peak.adducts = list(set([a.name for a in peak.adducts]))

                if not hasattr(peak, "Ms"):
                    setattr(peak, "Ms", [])

                peak.Ms = []
                for assignedAdduct in peak.adducts:
                    if assignedAdduct in adductsDict.keys():
                        peak.Ms.append((peak.mz - adductsDict[assignedAdduct].mzoffset) / adductsDict[assignedAdduct].mCount / peak.loading)

    # data processing step 6: convolute different feature pairs into feature groups using the chromatographic
    # profiles of different metabolite ions
    def groupFeaturePairsUntargetedAndWriteToDB(self, chromPeaks, mzxml, tracer, tracerID, reportFunction=None):
        try:
            db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

            nodes = {}
            correlations = {}

            allPeaks = {}
            for peak in chromPeaks:
                nodes[peak.id] = []
                allPeaks[peak.id] = peak
                peak.correlationsToOthers = []

            # sort by NPeakCenter once so the inner loop can break early on RT distance
            chromPeaks = sorted(chromPeaks, key=lambda p: p.NPeakCenter)

            ff_rows = []  # accumulated featurefeatures rows for a single bulk insert

            # compare all detected feature pairs at approximately the same retention time
            for piA in range(len(chromPeaks)):
                peakA = chromPeaks[piA]
                if reportFunction is not None:
                    reportFunction(
                        0.7 * piA / len(chromPeaks),
                        "Matching features (%d remaining)" % (len(chromPeaks) - piA),
                    )

                if peakA.id not in correlations.keys():
                    correlations[peakA.id] = {}

                for piB in range(piA + 1, len(chromPeaks)):
                    peakB = chromPeaks[piB]

                    # peaks are sorted by NPeakCenter; once the RT gap exceeds the threshold
                    # all remaining peaks are even further away — safe to stop early
                    if peakB.NPeakCenter - peakA.NPeakCenter >= self.peakCenterError:
                        break

                    if peakB.id not in correlations.keys():
                        correlations[peakB.id] = {}

                    if peakA.mz < peakB.mz:
                        if True:  # RT check already handled by the sorted early-exit above
                            bmin = int(
                                max(
                                    0,
                                    mean(
                                        [
                                            peakA.NPeakCenter - 1 * peakA.NBorderLeft,
                                            peakB.NPeakCenter - 1 * peakB.NBorderLeft,
                                            peakA.LPeakCenter - 1 * peakA.LBorderLeft,
                                            peakB.LPeakCenter - 1 * peakB.LBorderLeft,
                                        ]
                                    ),
                                )
                            )
                            bmax = int(
                                min(
                                    len(peakB.NXICSmoothed) - 1,
                                    mean(
                                        [
                                            peakB.NPeakCenter + 1 * peakB.NBorderRight,
                                            peakA.NPeakCenter + 1 * peakA.NBorderRight,
                                            peakB.LPeakCenter + 1 * peakB.LBorderRight,
                                            peakA.LPeakCenter + 1 * peakA.LBorderRight,
                                        ]
                                    ),
                                )
                            )

                            pb = corr(
                                peakA.NXICSmoothed[bmin:bmax],
                                peakB.NXICSmoothed[bmin:bmax],
                            )

                            if str(pb) == "nan":
                                pb = -1

                            correlations[peakA.id][peakB.id] = pb
                            correlations[peakB.id][peakA.id] = pb

                            silRatiosA = peakA.silRatios.silRatios
                            silRatiosB = peakB.silRatios.silRatios

                            meanSilRatioA = weightedMean(silRatiosA, peakA.silRatios.peakNInts)
                            meanSilRatioB = weightedMean(silRatiosB, peakB.silRatios.peakNInts)

                            silRatiosFold = 0
                            silRatiosSD = 1

                            try:
                                silRatiosFold = max(meanSilRatioA, meanSilRatioB) / min(meanSilRatioA, meanSilRatioB)
                                silRatiosSD = weightedSd(
                                    [abs(r - meanSilRatioA) / meanSilRatioA for r in silRatiosA if min(r, meanSilRatioA) > 0] + [abs(r - meanSilRatioB) / meanSilRatioB for r in silRatiosB if min(r, meanSilRatioB) > 0],
                                    peakA.silRatios.peakNInts + peakB.silRatios.peakNInts,
                                )

                                # check for similar chromatographic peak profile and similar native to labeled ratio
                                if pb >= self.minCorrelation and silRatiosFold <= 1 + max(0.25, 3 * silRatiosSD):
                                    nodes[peakA.id].append(peakB.id)
                                    nodes[peakB.id].append(peakA.id)

                                    peakB.correlationsToOthers.append(peakA.id)
                                    peakA.correlationsToOthers.append(peakB.id)

                            except Exception as e:
                                logging.error("Error while convoluting two feature pairs, skipping.. (%s)" % str(e))

                            try:
                                ff_rows.append({"fID1": peakA.id, "fID2": peakB.id, "corr": pb, "silRatioValue": silRatiosFold})
                            except Exception as e:
                                logging.error("Error while convoluting two feature pairs, skipping.. (%s)" % str(e))
                                ff_rows.append({"fID1": peakA.id, "fID2": peakB.id, "corr": 0, "silRatioValue": 0})

            # bulk-insert all featurefeatures rows at once instead of one per pair
            if ff_rows:
                _ff_schema = db_con.tables["featurefeatures"].schema
                _ff_new = pl.DataFrame(ff_rows, schema=_ff_schema)
                db_con.tables["featurefeatures"] = pl.concat([db_con.tables["featurefeatures"], _ff_new], how="vertical")

            self.postMessageToProgressWrapper("text", "%s: Convoluting feature groups" % tracer.name)

            for peak in chromPeaks:
                delattr(peak, "NXIC")
                delattr(peak, "LXIC")
                delattr(peak, "times")

            for k in nodes.keys():
                nodes[k] = list(set(nodes[k]))

            # get subgraphs from the feature pair graph. Each subgraph represents one convoluted
            # feature group
            tGroups = getSubGraphs(nodes)

            def splitGroupWithHCA(tGroup, correlations):
                # if 1 or two feature pairs are in a group, automatically use them (no further splitting possible)
                if len(tGroup) <= 2:
                    return [tGroup]

                # construct 2-dimensional data matrix
                data = []
                for tG1 in tGroup:
                    t = []
                    for tG2 in tGroup:
                        if tG1 in correlations.keys() and tG2 in correlations[tG1].keys():
                            t.append(correlations[tG1][tG2])
                        elif tG1 == tG2:
                            t.append(1.0)
                        else:
                            t.append(0.0)
                    data.append(t)

                # calculate HCA tree using the correlations between all feature pairs
                hc = HCA_general.HCA_generic()
                tree = hc.generateTree(objs=data, ids=tGroup)
                # hc.plotTree(tree)
                # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

                # split the HCA tree in sub-clusters
                def checkSubCluster(tree, hca, corrThreshold, cutOffMinRatio):
                    if isinstance(tree, HCA_general.HCALeaf):
                        return False
                    elif isinstance(tree, HCA_general.HCAComposite):
                        corrsLKid = hca.link(tree.getLeftKid())
                        corrs = hca.link(tree)
                        corrsRKid = hca.link(tree.getRightKid())

                        aboveThresholdLKid = sum([corr > corrThreshold for corr in corrsLKid])
                        aboveThreshold = sum([corr > corrThreshold for corr in corrs])
                        aboveThresholdRKid = sum([corr > corrThreshold for corr in corrsRKid])

                        # print(aboveThresholdLKid, aboveThreshold, aboveThresholdRKid)

                    if (aboveThresholdLKid * 1.0 / len(corrs)) >= cutOffMinRatio and (aboveThreshold * 1.0 / len(corrs)) >= cutOffMinRatio and (aboveThresholdRKid * 1.0 / len(corrs)) >= cutOffMinRatio:
                        return False
                    else:
                        return True

                # subClusts=hc.splitTreeWithCallbackBottomUp(tree,
                #                                   CallBackMethod(_target=checkSubCluster, corrThreshold=self.minCorrelation, cutOffMinRatio=self.minCorrelationConnections).getRunMethod())
                subClusts = hc.splitTreeWithCallback(
                    tree,
                    CallBackMethod(
                        _target=checkSubCluster,
                        corrThreshold=self.minCorrelation,
                        cutOffMinRatio=self.minCorrelationConnections,
                    ).getRunMethod(),
                    recursive=False,
                )

                # convert the subclusters into arrays of feature pairs belonging to the same metabolite
                return [[leaf.getID() for leaf in subClust.getLeaves()] for subClust in subClusts]

            groups = []
            done = 0
            for tGroup in tGroups:
                # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
                cGroups = [tGroup]
                while len(cGroups) > 0:
                    # print("HCA with", len(cGroups))
                    gGroup = cGroups.pop(0)

                    ## skip expensive HCA splitting for very large groups — keep them as-is
                    if len(gGroup) > 100:
                        groups.append(gGroup)
                        continue

                    sGroups = splitGroupWithHCA(gGroup, correlations)
                    # print("first group", len(gGroup), gGroup, "split into", sGroups)

                    if len(sGroups) == 1:
                        groups.append(sGroups[0])
                    else:
                        cGroups.extend(sGroups)

                done = done + 1
                self.postMessageToProgressWrapper(
                    "text",
                    "%s: Convoluting feature groups (%d/%d done)" % (tracer.name, done, len(tGroups)),
                )

            if True:
                done = 0
                for gi in range(len(groups)):
                    group = groups[gi]
                    if reportFunction is not None:
                        reportFunction(
                            0.7 + 0.3 * piA / len(chromPeaks),
                            "Searching for relationships (%d remaining)" % (len(groups) - gi),
                        )

                    # first, search for adduct relationships (same number of carbon atoms) in each convoluted
                    # feature group

                    peaksInGroup = {}
                    for a in group:
                        peaksInGroup[a] = allPeaks[a]

                        temp = []
                        for j in range(len(peaksInGroup[a].correlationsToOthers)):
                            if peaksInGroup[a].correlationsToOthers[j] in group:
                                temp.append(peaksInGroup[a].correlationsToOthers[j])

                        peaksInGroup[a].correlationsToOthers = temp

                    self.annotateChromPeaks(group, peaksInGroup)  # store feature pair annotation in the database

                    done = done + 1
                    self.postMessageToProgressWrapper(
                        "text",
                        "%s: Annotating feature groups (%d/%d done)" % (tracer.name, done, len(groups)),
                    )

            # collect all per-peak update data and apply in a single batch join-based update
            _update_ids = []
            _update_adducts = []
            _update_fDesc = []
            _update_corrToOthers = []
            _update_heteroAtoms = []

            for peak in chromPeaks:
                adds = countEntries(peak.adducts)
                peak.adducts = list(adds.keys())

                _update_ids.append(peak.id)
                _update_adducts.append(base64.b64encode(dumps(peak.adducts)).decode("utf-8"))
                _update_fDesc.append(base64.b64encode(dumps(peak.fDesc)).decode("utf-8"))
                _update_corrToOthers.append(base64.b64encode(dumps(peak.correlationsToOthers)).decode("utf-8"))
                _update_heteroAtoms.append(base64.b64encode(dumps(peak.heteroAtomsFeaturePairs)).decode("utf-8"))

            _upd_df = pl.DataFrame(
                {
                    "id": _update_ids,
                    "_adducts": _update_adducts,
                    "_fDesc": _update_fDesc,
                    "_corrToOthers": _update_corrToOthers,
                    "_heteroAtoms": _update_heteroAtoms,
                }
            )

            # single join-based update for chromPeaks
            db_con.tables["chromPeaks"] = (
                db_con.tables["chromPeaks"]
                .join(_upd_df, on="id", how="left")
                .with_columns(
                    pl.coalesce([pl.col("_adducts"), pl.col("adducts")]).alias("adducts"),
                    pl.coalesce([pl.col("_fDesc"), pl.col("fDesc")]).alias("fDesc"),
                    pl.coalesce([pl.col("_corrToOthers"), pl.col("correlationsToOthers")]).alias("correlationsToOthers"),
                    pl.coalesce([pl.col("_heteroAtoms"), pl.col("heteroAtomsFeaturePairs")]).alias("heteroAtomsFeaturePairs"),
                )
                .drop(["_adducts", "_fDesc", "_corrToOthers", "_heteroAtoms"])
            )

            # single join-based update for allChromPeaks (no correlationsToOthers column there)
            db_con.tables["allChromPeaks"] = (
                db_con.tables["allChromPeaks"]
                .join(_upd_df.select(["id", "_adducts", "_fDesc", "_heteroAtoms"]), on="id", how="left")
                .with_columns(
                    pl.coalesce([pl.col("_adducts"), pl.col("adducts")]).alias("adducts"),
                    pl.coalesce([pl.col("_fDesc"), pl.col("fDesc")]).alias("fDesc"),
                    pl.coalesce([pl.col("_heteroAtoms"), pl.col("heteroAtomsFeaturePairs")]).alias("heteroAtomsFeaturePairs"),
                )
                .drop(["_adducts", "_fDesc", "_heteroAtoms"])
            )

            # store feature group in the database
            for group in sorted(
                groups,
                key=lambda x: sum([allPeaks[p].NPeakCenterMin / 60.0 for p in x]) / len(x),
            ):
                db_con.insert_row("featureGroups", {"id": self.curFeatureGroupID, "featureName": "fg_%d" % self.curFeatureGroupID, "tracer": tracerID})
                groupMeanElutionIndex = 0

                hasPos = False
                hasNeg = False

                for p in sorted(group, key=lambda x: allPeaks[x].mz):
                    hasPos = hasPos or allPeaks[p].ionMode == "+"
                    hasNeg = hasNeg or allPeaks[p].ionMode == "-"

                    groupMeanElutionIndex += allPeaks[p].NPeakCenter

                    allPeaks[p].fGroupID = self.curFeatureGroupID
                    db_con.insert_row("featureGroupFeatures", {"fID": p, "fDesc": "", "fGroupID": self.curFeatureGroupID})

                groupMeanElutionIndex = groupMeanElutionIndex / len(group)

                # store one positve and one negative ionisation mode MS scan in the database for
                # later visualisation (one for each convoluted feature group)
                if hasPos:
                    scan = mzxml.getIthMS1Scan(int(groupMeanElutionIndex), self.positiveScanEvent)

                    db_con.insert_row("massspectrum", {"mID": self.curMassSpectrumID, "fgID": self.curFeatureGroupID, "time": scan.retention_time, "mzs": ";".join([str(u) for u in scan.mz_list]), "intensities": ";".join([str(u) for u in scan.intensity_list]), "ionMode": "+"})
                    self.curMassSpectrumID += 1
                if hasNeg:
                    scan = mzxml.getIthMS1Scan(int(groupMeanElutionIndex), self.negativeScanEvent)

                    db_con.insert_row("massspectrum", {"mID": self.curMassSpectrumID, "fgID": self.curFeatureGroupID, "time": scan.retention_time, "mzs": ";".join([str(u) for u in scan.mz_list]), "intensities": ";".join([str(u) for u in scan.intensity_list]), "ionMode": "-"})
                    self.curMassSpectrumID += 1

                self.curFeatureGroupID += 1

            db_con.commit()
            self.printMessage(
                "Feature grouping done. " + str(len(groups)) + " feature groups",
                type="info",
            )

            db_con.commit()
            db_con.close()

        except Exception as ex:
            traceback.print_exc()

            self.printMessage("Error: %s" % (str(ex)), type="error")
            self.postMessageToProgressWrapper("failed", self.pID)

    # store one MS scan for each detected feature pair in the database
    def writeMassSpectraToDB(self, chromPeaks, mzxml, reportFunction=None):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        massSpectraWrittenPos = {}
        massSpectraWrittenNeg = {}
        for pi in range(len(chromPeaks)):
            peak = chromPeaks[pi]
            if reportFunction is not None:
                reportFunction(
                    1.0 * pi / len(chromPeaks),
                    "%d feature pairs remaining" % (len(chromPeaks) - pi),
                )

            scanEvent = ""
            iMode = ""
            massSpectraWritten = None
            if peak.ionMode == "+":
                iMode = "+"
                scanEvent = self.positiveScanEvent
                massSpectraWritten = massSpectraWrittenPos
            elif peak.ionMode == "-":
                iMode = "-"
                scanEvent = self.negativeScanEvent
                massSpectraWritten = massSpectraWrittenNeg

            if peak.NPeakCenter not in massSpectraWritten.keys():
                scan = mzxml.getIthMS1Scan(peak.NPeakCenter, scanEvent)

                db_con.insert_row("massspectrum", {"mID": self.curMassSpectrumID, "fgID": -1, "time": scan.retention_time, "mzs": ";".join([str(u) for u in scan.mz_list]), "intensities": ";".join([str(u) for u in scan.intensity_list]), "ionMode": iMode})

                massSpectraWritten[peak.NPeakCenter] = self.curMassSpectrumID
                self.curMassSpectrumID = self.curMassSpectrumID + 1

            # Update massSpectrumID in chromPeaks
            db_con.tables["chromPeaks"] = db_con.tables["chromPeaks"].with_columns(pl.when(pl.col("id") == peak.id).then(pl.lit(massSpectraWritten[peak.NPeakCenter])).otherwise(pl.col("massSpectrumID")).alias("massSpectrumID"))

        db_con.commit()
        db_con.close()

    ## write a new featureML file
    def writeResultsToFeatureML(self, forFile):
        db_con = PolarsDB(forFile + getDBSuffix(), format=getDBFormat())

        features = []

        # Get chromPeaks data directly from Polars DataFrame
        df = db_con.tables.get("chromPeaks", pl.DataFrame())
        for row in df.to_dicts():
            chromPeak = ChromPeakPair()
            chromPeak.id = row["id"]
            chromPeak.mz = row["mz"]
            chromPeak.lmz = row["lmz"]
            chromPeak.xCount = row["xcount"]
            chromPeak.loading = row["Loading"]
            chromPeak.NPeakCenterMin = row["NPeakCenterMin"]
            chromPeak.ionMode = row["ionMode"]

            b = Bunch(
                id=chromPeak.id,
                ogroup="-1",
                mz=chromPeak.mz,
                rt=chromPeak.NPeakCenterMin,
                Xn=chromPeak.xCount,
                lmz=chromPeak.lmz,
                charge=chromPeak.loading,
                name=chromPeak.id,
                ionMode=chromPeak.ionMode,
            )
            features.append(b)

        exportAsFeatureML.writeFeatureListToFeatureML(features, forFile + ".featureML", ppmPM=self.ppm, rtPM=0.25 * 60)

    # store the detected feature pairs in a new mzXML file. Only those MS peaks will be included, which contribute to
    # the chromatographic peaks of a valid feature pair
    def writeResultsToNewMZXMLIntermediateObject(self, mzxml, newMZXMLData, chromPeaks):
        for peak in chromPeaks:
            # writeMZXML: 0001/1: 12C  0010/2: 12C-Iso  0100/4: 13C-Iso  1000/8: 13C

            if peak.ionMode == "+":
                scanEv = self.positiveScanEvent
            elif peak.ionMode == "-":
                scanEv = self.negativeScanEvent

            if self.writeMZXML & 1:
                scans, times, scanIds = mzxml.getArea(
                    peak.NPeakCenter - peak.NBorderLeft,
                    peak.NPeakCenter + peak.NBorderRight,
                    peak.mz,
                    self.chromPeakPPM,
                    scanEv,
                )

                for w in range(len(scans)):
                    curScan = scans[w]
                    scanid = scanIds[w]
                    if scanid not in newMZXMLData:
                        newMZXMLData[scanid] = Bunch(mzs=[], ints=[], rt=mzxml.getScanByID(scanid).retention_time)
                    for mz, inte in curScan:
                        newMZXMLData[scanid].mzs.append(mz)
                        newMZXMLData[scanid].ints.append(inte)
            if self.writeMZXML & 2:
                scans, times, scanIds = mzxml.getArea(
                    peak.NPeakCenter - peak.NBorderLeft,
                    peak.NPeakCenter + peak.NBorderRight,
                    peak.mz + self.xOffset / peak.loading,
                    self.chromPeakPPM,
                    scanEv,
                )

                for w in range(len(scans)):
                    curScan = scans[w]
                    scanid = scanIds[w]
                    if scanid not in newMZXMLData:
                        newMZXMLData[scanid] = Bunch(mzs=[], ints=[], rt=mzxml.getScanByID(scanid).retention_time)
                    for mz, inte in curScan:
                        newMZXMLData[scanid].mzs.append(mz)
                        newMZXMLData[scanid].ints.append(inte)
            if self.writeMZXML & 4:
                scans, times, scanIds = mzxml.getArea(
                    peak.NPeakCenter - peak.NBorderLeft,
                    peak.NPeakCenter + peak.NBorderRight,
                    peak.mz + (peak.xCount - 1) * self.xOffset / peak.loading,
                    self.chromPeakPPM,
                    scanEv,
                )

                for w in range(len(scans)):
                    curScan = scans[w]
                    scanid = scanIds[w]
                    if scanid not in newMZXMLData:
                        newMZXMLData[scanid] = Bunch(mzs=[], ints=[], rt=mzxml.getScanByID(scanid).retention_time)
                    for mz, inte in curScan:
                        newMZXMLData[scanid].mzs.append(mz)
                        newMZXMLData[scanid].ints.append(inte)
            if self.writeMZXML & 8:
                scans, times, scanIds = mzxml.getArea(
                    peak.NPeakCenter - peak.NBorderLeft,
                    peak.NPeakCenter + peak.NBorderRight,
                    peak.mz + peak.xCount * self.xOffset / peak.loading,
                    self.chromPeakPPM,
                    scanEv,
                )

                for w in range(len(scans)):
                    curScan = scans[w]
                    scanid = scanIds[w]
                    if scanid not in newMZXMLData:
                        newMZXMLData[scanid] = Bunch(mzs=[], ints=[], rt=mzxml.getScanByID(scanid).retention_time)
                    for mz, inte in curScan:
                        newMZXMLData[scanid].mzs.append(mz)
                        newMZXMLData[scanid].ints.append(inte)

    # helper method if more than one tracer substance is used
    def writeIntermediateMZXMLDataToNewMZXMLFile(self, mzxml, newMZXMLData):
        try:
            self.printMessage("Writing MzXML file..", type="info")
            if ".mzxml" in self.file.lower():
                toFile = self.file[0 : self.file.lower().rfind(".mzxml")] + ".proc.mzXML"
                mzxml.resetMZData(self.file, toFile, newMZXMLData)
            elif ".mzml" in self.file.lower() and False:  ## mzml export currently not supported
                toFile = self.file[0 : self.file.lower().rfind(".mzml")] + "proc.mzML"
                self.printMessage("Only mzXML files can be written", type="error")
            else:
                return RuntimeError("Invalid file. Cannot write new data")

        except Exception as ex:
            self.printMessage("Cannot write MzXML file.. (%s)" % ex, type="error")

    # stores the TICs of the LC-HRMS data in the database
    def writeTICsToDB(self, mzxml, scanEvents):
        db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

        ## save TICs
        ## save mean, sd scan time
        i = 1
        for pol, scanEvent in scanEvents.items():
            if scanEvent != "None":
                TIC, times, scanIds = mzxml.getTIC(filterLine=scanEvent)

                # Insert into tics table
                db_con.insert_row("tics", {"id": i, "polarity": pol, "scanevent": scanEvent, "times": ";".join([str(t) for t in times]), "intensities": ";".join(["%.0f" % t for t in TIC])})
                i = i + 1

                scanTimes = [times[i + 1] - times[i] for i in range(len(times) - 1)]

                # Insert stats
                db_con.insert_row("stats", {"key": "MeanScanTime_%s" % pol, "value": str(mean(scanTimes))})
                db_con.insert_row("stats", {"key": "SDScanTime_%s" % pol, "value": str(sd(scanTimes))})
                db_con.insert_row("stats", {"key": "NumberOfScans_%s" % pol, "value": str(len(times))})
                db_con.insert_row("stats", {"key": "TotalNumberOfSignals_%s" % pol, "value": str(mzxml.getSignalCount(filterLine=scanEvent))})

                minInt, maxInt, avgInt = mzxml.getMinMaxAvgSignalIntensities(filterLine=scanEvent)
                db_con.insert_row("stats", {"key": "MinSignalInt_%s" % pol, "value": str(minInt)})
                db_con.insert_row("stats", {"key": "MaxSignalInt_%s" % pol, "value": str(maxInt)})
                db_con.insert_row("stats", {"key": "AvgSignalInt_%s" % pol, "value": str(avgInt)})

        db_con.commit()
        db_con.close()

    # method, which is called by the multiprocessing module to actually process the LC-HRMS data
    def findIsoPairs(self):
        try:
            start = time.time()

            # region Initialize data processing pipeline
            ######################################################################################

            self.postMessageToProgressWrapper("start")
            self.postMessageToProgressWrapper("max", 100.0)
            self.postMessageToProgressWrapper("value", 0.0)
            self.postMessageToProgressWrapper("text", "Initialising")

            self.BL = Baseline.Baseline()

            self.curPeakId = 1
            self.curEICId = 1
            self.curMZId = 1
            self.curMZBinId = 1
            self.curFeatureGroupID = 1
            self.curMassSpectrumID = 1
            # endregion

            # region Create results database
            ######################################################################################
            self.postMessageToProgressWrapper("text", "Creating results DB")

            if os.path.exists(self.file + getDBSuffix()) and os.path.isfile(self.file + getDBSuffix()):
                os.remove(self.file + getDBSuffix())
            db_con = PolarsDB(self.file + getDBSuffix(), format=getDBFormat())

            self.writeConfigurationToDB(db_con)

            db_con.close()
            # endregion

            # region Parse mzXML file
            ######################################################################################

            self.postMessageToProgressWrapper("text", "Parsing chromatogram file")

            mzxml = self.parseMzXMLFile()
            newMZXMLData = {}

            self.writeTICsToDB(mzxml, {"+": self.positiveScanEvent, "-": self.negativeScanEvent})

            # endregion

            # Start calculation
            ######################################################################################

            self.postMessageToProgressWrapper("text", "Starting data processing")
            tracerProgressWidth = 100.0 / 1

            tracerNum = 1

            # region Process configured tracer
            ######################################################################################

            tracer = self.configuredTracer

            curTracerProgress = tracerNum / 1

            tracerID = 0
            if self.metabolisationExperiment:
                if tracer is None:
                    self.printMessage("Metabolisation experiment requires a configured tracer, but none was available in the identify method.", type="error")
                    raise Exception("Metabolisation experiment requires a configured tracer, but none was available in the identify method.")

                tracerID = tracer.id
                self.printMessage("Tracer: %s" % tracer.name, type="info")

                ##################################################################################################
                # Attention: delta mz for one labelling atom is always saved in the member variable self.xOffset #
                ##################################################################################################

                self.xOffset = getIsotopeMass(tracer.isotopeB)[0] - getIsotopeMass(tracer.isotopeA)[0]

            else:
                # Full metabolome labelling experiment
                pass
            # endregion

            # region 1. Find 12C 13C partners in the mz dimension (0-25%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "Extracting signal pairs")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0 * tracerProgressWidth + 0.25 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper("text", "Extracting signal pairs (%s)" % text)

            mzs, negFound, posFound = self.findSignalPairs(curTracerProgress, mzxml, tracer, reportFunction)
            self.writeSignalPairsToDB(mzs, mzxml, tracerID)

            self.printMessage(
                "Extracting signal pairs done. pos: %d neg: %d mzs (including mismatches)" % (posFound, negFound),
                type="info",
            )
            self.postMessageToProgressWrapper("text", "Extracting signal pairs done. pos: %d neg: %d mzs (including mismatches)" % (posFound, negFound), force=True)
            # endregion

            # region 2. Cluster found mz values according to mz value and number of x atoms (25-35%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0.25 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "Clustering found signal pairs")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0.25 * tracerProgressWidth + 0.1 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper(
                    "text",
                    "Clustering found signal pairs (%s)" % text,
                )

            mzbins = self.clusterFeaturePairs(mzs, reportFunction)
            self.writeFeaturePairClustersToDB(mzbins)
            mzbins = self.removeImpossibleFeaturePairClusters(mzbins)

            self.printMessage(
                "Clustering found signal pairs done. pos: %d neg: %d mz bins (including mismatches)" % (len(mzbins["+"]), len(mzbins["-"])),
                type="info",
            )
            # endregion

            # region 3. Extract chromatographic peaks (35-65%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0.35 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "%s: Separating feature pairs" % tracer.name)

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0.35 * tracerProgressWidth + 0.3 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper("text", "Separating feature pairs (%s)" % text)

            chromPeaks = self.findChromatographicPeaksAndWriteToDB(mzbins, mzxml, tracerID, reportFunction)

            self.printMessage(
                "Separating feature pairs done. pos: %d neg: %d chromatographic peaks (including mismatches)"
                % (
                    len([c for c in chromPeaks if c.ionMode == "+"]),
                    len([c for c in chromPeaks if c.ionMode == "-"]),
                ),
                type="info",
            )
            # endregion

            # region 4. Remove isotopolog feature pairs and other false positive findings (65-70%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0.65 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "Removing false positive feature pairs")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0.65 * tracerProgressWidth + 0.05 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper(
                    "text",
                    "Removing false positive feature pairs (%s)" % text,
                )

            self.removeFalsePositiveFeaturePairsAndUpdateDB(chromPeaks, reportFunction)

            self.printMessage(
                "Removing false positive feature pairs done. %d chromatographic peaks" % len(chromPeaks),
                type="info",
            )
            # endregion

            # region 5. Search for hetero atoms (70-75%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0.7 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "Searching for hetero atoms")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0.7 * tracerProgressWidth + 0.05 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper("text", "Annotating feature pairs (%s)" % text)

            self.annotateFeaturePairs(chromPeaks, mzxml, tracer, reportFunction)
            # endregion

            # region 6. Group feature pairs untargeted using chromatographic peak shape into feature groups (75-80%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0.75 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "Grouping feature pairs")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0.75 * tracerProgressWidth + 0.05 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper("text", "Grouping feature pairs (%s)" % text)

            self.groupFeaturePairsUntargetedAndWriteToDB(chromPeaks, mzxml, tracer, tracerID, reportFunction)
            # endregion

            # region 7. Extract mass spectra for feature pairs and feature groups (80-95%)
            ######################################################################################

            self.postMessageToProgressWrapper("value", curTracerProgress + 0.8 * tracerProgressWidth)
            self.postMessageToProgressWrapper("text", "Extracting mass spectra to DB")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper(
                    "value",
                    curTracerProgress + 0.8 * tracerProgressWidth + 0.15 * curVal * tracerProgressWidth,
                )
                self.postMessageToProgressWrapper(
                    "text",
                    "Extracting mass spectra to DB (%s)" % (text),
                )

            self.writeMassSpectraToDB(chromPeaks, mzxml, reportFunction)

            # Log time used for processing of individual files
            elapsed = (time.time() - start) / 60.0
            hours = ""
            if elapsed >= 60.0:
                if elapsed < 120.0:
                    hours = "1 hour "
                else:
                    hours = "%d hours " % (elapsed // 60)
            mins = "%.2f min(s)" % (elapsed % 60.0)

            self.printMessage(
                "Calculations finished (%s%s).." % (hours, mins),
                type="info",
            )
            # endregion

            # region 7b. Calculate isotopolog ratios for each feature pair
            ######################################################################################

            self.postMessageToProgressWrapper("text", "Calculating isotopolog ratios")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper("text", "Calculating isotopolog ratios (%s)" % text)

            self.calculateIsotopologRatiosForFeaturePairs(chromPeaks, mzxml, reportFunction)

            self.printMessage(
                "Isotopolog ratio calculation done.",
                type="info",
            )
            # endregion

            # region 7c. Calculate isotopolog enrichment for each feature pair
            ######################################################################################

            self.postMessageToProgressWrapper("text", "Calculating isotopolog enrichment")

            def reportFunction(curVal, text):
                self.postMessageToProgressWrapper("text", "Calculating isotopolog enrichment (%s)" % text)

            self.calculateIsotopologEnrichmentForFeaturePairs(chromPeaks, mzxml, reportFunction)

            self.printMessage(
                "Isotopolog enrichment calculation done.",
                type="info",
            )
            # endregion

            # region 8. Write results to files (95-100%, without progress indicator)
            ######################################################################################

            # W.1 Save results to new MzXML file (intermediate step) (95-100%)

            if False and self.writeMZXML:
                self.postMessageToProgressWrapper("value", curTracerProgress + 0.95 * tracerProgressWidth)
                self.postMessageToProgressWrapper("text", "Writing results to mzXML..")

                self.writeResultsToNewMZXMLIntermediateObject(mzxml, newMZXMLData, chromPeaks)
            # endregion

            self.postMessageToProgressWrapper("text", "Writing results to mzXML..")

            # region W.2 Write results to TSV File
            ##########################################################################################
            if False and self.writeFeatureML:
                self.postMessageToProgressWrapper("text", "Writing results to featureML..")

                self.writeResultsToFeatureML(self.file)
            # endregion

            # region W.4 Save all results to new MzXML file
            ##########################################################################################
            if False and self.writeMZXML:
                self.postMessageToProgressWrapper("text", "Writing results to new mzXML file")

                self.writeIntermediateMZXMLDataToNewMZXMLFile(mzxml, newMZXMLData)
            # endregion

            mzxml.freeMe()
            self.postMessageToProgressWrapper("end")

        except Exception as ex:
            self.printMessage("Error: %s" % (str(ex)), type="error")
            self.postMessageToProgressWrapper("failed")
            traceback.print_exc()

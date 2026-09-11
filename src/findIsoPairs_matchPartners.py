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


from copy import deepcopy
import traceback
from .formulaTools import formulaTools
from .utils import Bunch, getNormRatio

maxSub = 5


def calculate_theoretical_ratios(isotopic_enrichment, Xn_max, maxSub):
    """
    calculates the theoretical isotopolog distribution for a given isotopic enrichment and number of atoms
    isotopic_enrichment: isotopic enrichment of the isotope element (e.g. 0.9893 for 12C)
    Xn_max: maximum number of atoms that can be exchanged (e.g. 3 for 3 carbon atoms)
    maxSub: maximum number of substitutions
    """
    ret = []
    ret.append([])
    for Xn in range(1, Xn_max + 1):
        cur = []
        for cur_exchange_number in range(0, min(Xn, maxSub) + 1):
            cur.append(getNormRatio(isotopic_enrichment, Xn, cur_exchange_number))
        ret.append(cur)
    return ret


# results object (ion signal pairs)
class mzFeature:
    def __init__(self, id, mz, lmz, tmz, xCount, scanIndex, scanid, scantime, loading, nIntensity, lIntensity, ionMode):
        self.id = id
        self.mz = mz
        self.lmz = lmz
        self.tmz = tmz
        self.xCount = xCount
        self.scanIndex = scanIndex
        self.scanid = scanid
        self.scantime = scantime
        self.loading = loading
        self.nIntensity = nIntensity
        self.lIntensity = lIntensity
        self.ionMode = ionMode

    def __str__(self):
        return "mz: %.4f, lmz: %.4f, xCount: %d, charge: %d, NInt: %.1f, LInt: %.1f, ionMode: %s, scanIndex: %d" % (
            self.mz,
            self.lmz,
            self.xCount,
            self.loading,
            self.nIntensity,
            self.lIntensity,
            self.ionMode,
            self.scanIndex,
        )


## calculates all possible combinations of labeling elements
## required for double labeling
def getCombinationsOfLabel(
    useElems,
    labelingElements,
    minLabelingAtoms,
    maxLabelingAtoms,
    used=None,
    startAt=0,
    ind=2,
):
    combs = []
    if used is None:
        used = {}

    x = sum(used.values())
    if minLabelingAtoms <= x <= maxLabelingAtoms:
        b = Bunch(
            atoms=deepcopy(used),
            atomsCount=sum([used[e] for e in used.keys()]),
            mzdelta=sum([(labelingElements[e].massLabeled - labelingElements[e].massNative) * used[e] for e in used.keys()]),
        )
        combs.append(b)

    if startAt < len(useElems):
        for cStartAt in range(startAt, len(useElems)):
            e = useElems[cStartAt]
            for i in range(labelingElements[e].minXn, labelingElements[e].maxXn + 1):
                c = deepcopy(used)
                c[e] = i

                combs.extend(
                    getCombinationsOfLabel(
                        useElems,
                        labelingElements,
                        minLabelingAtoms,
                        maxLabelingAtoms,
                        c,
                        cStartAt + 1,
                        ind=ind + 2,
                    )
                )

    return combs


# detects in each recorded MS scan (lvl. 1) isotope patterns originating from a native and a (partially) labelled
# metabolite / biotransformation product. It is important, that all atoms that can be labelled are actually
# labelled and have the same chance to be labelled. Thus, fluxomics applications or such isotope patterns
# are in general not supported. They can, however, be detected if not isotopolog verification step is
# used (not recommended)
def matchPartners(
    mzXMLData,
    forFile,
    labellingElement,
    useCIsotopePatternValidation,
    intensityThres,
    isotopologIntensityThres,
    maxLoading,
    xCounts,
    xOffset,
    ppm,
    intensityErrorN,
    intensityErrorL,
    purityN,
    purityL,
    startTime,
    stopTime,
    filterLine,
    ionMode,
    peakCountLeft,
    peakCountRight,
    lowAbundanceIsotopeCutoff,
    metabolisationExperiment,
    checkRatio,
    minRatio,
    maxRatio,
    scanIndexOffset=0,
    reportFunction=None,
    writeExtendedDiagnostics=True,
):
    scan_rt_range = stopTime - startTime

    c13c12_offset = 1.00335484  # mass difference between 12C and 13C
    ori_Xoffset = xOffset

    detected_signal_pairs = []
    fT = formulaTools()

    if labellingElement == "C":
        useCIsotopePatternValidation = 0
        # Carbon validation is the only possible method
        # when the labelling element is carbon (to some extend that's logical)

    # substitution arrays for checking the number of carbon atoms
    theoretical_ratios_native = calculate_theoretical_ratios(purityN, max(xCounts) + 1, maxSub)  # native metabolite
    theoretical_ratios_labeled = calculate_theoretical_ratios(purityL, max(xCounts) + 1, maxSub)  # labelled metabolite

    # beta-feature, support for two labeling elements at once (e.g., custom tracers)
    useDoubleLabelingCombinations = False
    if useDoubleLabelingCombinations:
        labelingElements = {}
        labelingElements["C"] = Bunch(
            nativeIsotope="12C",
            labelingIsotope="13C",
            massNative=12.0,
            massLabeled=13.00335,
            isotopicEnrichmentNative=0.9893,
            isotopicEnrichmentLabeled=0.995,
            minXn=1,
            maxXn=3,
        )
        labelingElements["H"] = Bunch(
            nativeIsotope="1H",
            labelingIsotope="2H",
            massNative=1.0078250,
            massLabeled=2.0141018,
            isotopicEnrichmentNative=0.9999,
            isotopicEnrichmentLabeled=0.96,
            minXn=0,
            maxXn=9,
        )

        ## combinations of labeling elements
        tempCombs = getCombinationsOfLabel(["C", "H"], labelingElements, 2, 12)
        combs = []
        for comb in tempCombs:
            ## remove implausible labeling combinations
            if True:
                if "C" in comb.atoms.keys() and comb.atoms["C"] == 1 and ("H" not in comb.atoms.keys() or ("H" in comb.atoms.keys() and comb.atoms["H"] in [1, 2, 3])):
                    combs.append(comb)
                if "C" in comb.atoms.keys() and comb.atoms["C"] == 2 and ("H" not in comb.atoms.keys() or ("H" in comb.atoms.keys() and comb.atoms["H"] in [1, 2, 3, 4, 5, 6])):
                    combs.append(comb)
                if "C" in comb.atoms.keys() and comb.atoms["C"] == 3 and ("H" not in comb.atoms.keys() or ("H" in comb.atoms.keys() and comb.atoms["H"] in [1, 2, 3, 4, 5, 6, 7, 8, 9])):
                    combs.append(comb)
        if useDoubleLabelingCombinations:
            print("The following labeling configurations are used:")
            for comb in combs:
                print(comb)

    # other parameters
    max_other_iso_intensity_ratio = 0.25

    # iterate over all MS scans (lvl. 1)
    cur_native_scan_index = 0
    next_signal_pair_id = 1
    for cur_scan_index in range(0, len(mzXMLData.MS1_list)):
        try:
            cur_scan = mzXMLData.MS1_list[cur_scan_index]
            cur_scan_detected_signal_pairs = []

            # check for correct filterline and scan time
            if cur_scan.filter_line == filterLine:
                if startTime <= (cur_scan.retention_time / 60.0) <= stopTime:
                    if reportFunction is not None:
                        reportFunction(
                            (cur_scan.retention_time / 60.0 - startTime) / scan_rt_range,
                            "RT %.2f, found patterns: %d" % (cur_scan.retention_time / 60.0, len(detected_signal_pairs)),
                        )

                    # Compute the scan used to search for labeled signals; may be offset relative to the native scan.
                    # Falls back to curScan when the offset puts the index out of bounds or on a different filter line.
                    # but make sure that the offset is scanIndexOffset scans of the correct filter_line
                    lab_scan_index = cur_scan_index
                    if scanIndexOffset != 0:
                        cur_scan_Index_Offset = 0
                        direction = 1 if scanIndexOffset >= 0 else -1
                        while lab_scan_index > 0:
                            if mzXMLData.MS1_list[lab_scan_index].filter_line == filterLine:
                                if cur_scan_Index_Offset == scanIndexOffset:
                                    break
                                else:
                                    cur_scan_Index_Offset += direction
                            lab_scan_index += direction

                    # use labeled scan, test if it is valid (correct filter line, within bounds)
                    labScanJ = lab_scan_index
                    if 0 <= labScanJ < len(mzXMLData.MS1_list) and mzXMLData.MS1_list[labScanJ].filter_line == filterLine:
                        labScan = mzXMLData.MS1_list[labScanJ]
                    else:
                        raise RuntimeError(f"ERROR: Could not find a suitable lab scan for scan index {cur_scan_index} with offset {scanIndexOffset}. ")

                    dontUsePeakIndices = []

                    # assume each peak to be a valid M (monoisotopic, native metabolite ion)
                    # and verify this assumption (search for (partially) labelled pendant)
                    for currentPeakIndex in range(0, len(cur_scan.mz_list)):
                        if currentPeakIndex not in dontUsePeakIndices:
                            isoM_mz = cur_scan.mz_list[currentPeakIndex]
                            isoM_int = cur_scan.intensity_list[currentPeakIndex]

                            cur_signal_detected_partners = []

                            # only consider peaks above the threshold
                            if isoM_int >= intensityThres:
                                skip_other_loadings = False

                                ## do not process peaks that are likely isotopologs
                                m_negative_C_isotopologs = []
                                for current_charge in range(maxLoading, 0, -1):
                                    iso = cur_scan.findMZ(isoM_mz - c13c12_offset / current_charge, ppm)
                                    iso = cur_scan.getMostIntensePeak(iso[0], iso[1])

                                    if iso != -1 and cur_scan.intensity_list[iso] > isoM_int * max_other_iso_intensity_ratio:
                                        m_negative_C_isotopologs.append(current_charge)
                                if len(m_negative_C_isotopologs) > 0:
                                    continue

                                possible_charges = []
                                ## figure out possible loadings
                                for current_charge in range(maxLoading, 0, -1):
                                    iso = cur_scan.findMZ(
                                        isoM_mz + c13c12_offset / current_charge,
                                        ppm,
                                        start=currentPeakIndex,
                                    )
                                    iso = cur_scan.getMostIntensePeak(iso[0], iso[1])

                                    if iso != -1:
                                        possible_charges.append(current_charge)
                                        break  ## skip other loadings

                                if len(possible_charges) == 0:
                                    possible_charges = [1]

                                for current_charge in possible_charges:
                                    if not skip_other_loadings:
                                        xOffset_charged = ori_Xoffset / float(current_charge)
                                        c13c12_offset_charged = c13c12_offset / float(current_charge)

                                        # C-isotope distribution validation for labelling with N, S, ... (useCValidation == 2)
                                        # The carbon distribution of both isotopologs is checked for equality
                                        # checks if the isotope patterns of M, M+1.. and M', M'+1.. are approximately the same
                                        # E.g. 15N-labelling
                                        # |    <--   Nn   -->    |
                                        # ||                     ||
                                        # |||                   .|||
                                        # Required for some labelling applications (e.g. S, N, Cl)
                                        # Requires: - a high resolution and separation of different isotoplogs (especially carbon)
                                        # EXPERIMENTAL: has not been tested with real data (not N or S labelled sample material
                                        #               was available)
                                        if not useDoubleLabelingCombinations and useCIsotopePatternValidation != 0:
                                            isoM_m1_index = cur_scan.findMZ(isoM_mz - c13c12_offset_charged, ppm * 2)
                                            isoM_m1_index = cur_scan.getMostIntensePeak(isoM_m1_index[0], isoM_m1_index[1])
                                            if isoM_m1_index != -1:
                                                obs_ratio_Mm1_to_M = cur_scan.intensity_list[isoM_m1_index] / isoM_int
                                                if obs_ratio_Mm1_to_M >= 0.5:
                                                    continue

                                            # When lowAbundanceIsotopeCutoff is enabled, also check M-1 abundance
                                            # to verify the detected peak is truly M and not an isotopolog artifact
                                            if lowAbundanceIsotopeCutoff and isoM_m1_index != -1:
                                                isoM_m1_Intensity = cur_scan.intensity_list[isoM_m1_index]
                                                if isoM_m1_Intensity > isoM_int:
                                                    # M-1 is more intense than M: likely not the monoisotopic peak
                                                    continue

                                            # find M+1 peak
                                            isoM_p1_index = cur_scan.findMZ(
                                                isoM_mz + c13c12_offset_charged,
                                                ppm,
                                                start=currentPeakIndex,
                                            )
                                            isoM_p1_index = cur_scan.getMostIntensePeak(isoM_p1_index[0], isoM_p1_index[1])
                                            if isoM_p1_index != -1 or peakCountLeft == 1 or lowAbundanceIsotopeCutoff:
                                                # test certain number of labelled carbon atoms

                                                for current_Cn in sorted(xCounts, reverse=True):
                                                    # find corresponding M' peak (search in labScan, which may be offset)
                                                    isoMP_index = labScan.findMZ(
                                                        isoM_mz + current_Cn * c13c12_offset_charged,
                                                        ppm,
                                                    )
                                                    isoMP_index = labScan.getMostIntensePeak(
                                                        isoMP_index[0],
                                                        isoMP_index[1],
                                                        intensityThres,
                                                    )
                                                    if isoMP_index != -1:
                                                        isoMP_mz = labScan.mz_list[isoMP_index]
                                                        isoMP_int = labScan.intensity_list[isoMP_index]

                                                        # find M'-1 peak
                                                        isoMP_m1_index = labScan.findMZ(
                                                            isoM_mz + current_Cn * c13c12_offset_charged - c13c12_offset_charged,
                                                            ppm * 2,
                                                        )
                                                        isoMP_m1_index = labScan.getMostIntensePeak(
                                                            isoMP_m1_index[0],
                                                            isoMP_m1_index[1],
                                                        )
                                                        if isoMP_m1_index != -1:
                                                            obs_ratio_Mm1_to_M = labScan.intensity_list[isoMP_m1_index] / labScan.intensity_list[isoMP_index]
                                                            if obs_ratio_Mm1_to_M >= 0.5:
                                                                continue

                                                        # (1.) check if M' and M ratio are as expected
                                                        if checkRatio:
                                                            rat = isoM_int / isoMP_int
                                                            if minRatio <= rat <= maxRatio:
                                                                pass  ## ratio check passed
                                                            else:
                                                                continue  ## ratio check not passed

                                                        ## no isotopolog verification needs to be performed
                                                        if peakCountLeft == 1 and peakCountRight == 1:
                                                            cur_signal_detected_partners.append(
                                                                mzFeature(
                                                                    id=len(cur_signal_detected_partners),
                                                                    mz=isoM_mz,
                                                                    lmz=isoMP_mz,
                                                                    tmz=current_Cn * c13c12_offset_charged,
                                                                    xCount=current_Cn,
                                                                    scanIndex=cur_native_scan_index,
                                                                    scanid=cur_scan.id,
                                                                    scantime=cur_scan.retention_time,
                                                                    loading=current_charge,
                                                                    nIntensity=isoM_int,
                                                                    lIntensity=isoMP_int,
                                                                    ionMode=ionMode,
                                                                )
                                                            )

                                                            skip_other_loadings = True
                                                            continue

                                                        # find M'+1 peak
                                                        isoMP_p1_index = labScan.findMZ(
                                                            isoM_mz + current_Cn * c13c12_offset_charged + c13c12_offset_charged,
                                                            ppm,
                                                        )
                                                        isoMP_p1_index = labScan.getMostIntensePeak(
                                                            isoMP_p1_index[0],
                                                            isoMP_p1_index[1],
                                                        )

                                                        # calculate the ratio of M+1/M (native, always in curScan)
                                                        isoPeakIntensity = cur_scan.intensity_list[isoM_p1_index]
                                                        if peakCountLeft == 1:
                                                            ratioN = None
                                                        elif peakCountLeft > 1 and lowAbundanceIsotopeCutoff and isoPeakIntensity <= isotopologIntensityThres:
                                                            ratioN = None
                                                        else:
                                                            ratioN = isoPeakIntensity / isoM_int

                                                        # calculate the ratio of M'+1/M' (labeled, in labScan)
                                                        isoLabPeakIntensity = labScan.intensity_list[isoMP_p1_index]
                                                        if peakCountRight == 1:
                                                            ratioL = None
                                                        elif peakCountRight > 1 and lowAbundanceIsotopeCutoff and isoLabPeakIntensity <= isotopologIntensityThres:
                                                            ratioL = None
                                                        else:
                                                            ratioL = isoLabPeakIntensity / isoMP_int
                                                        # 2. check if the observed M'+1/M' ratio and the M+1/M ratio are approximately equal
                                                        if (ratioN is not None and ratioL is not None) and abs(ratioN - ratioL) <= intensityErrorL:
                                                            cur_signal_detected_partners.append(
                                                                mzFeature(
                                                                    id=len(cur_signal_detected_partners),
                                                                    mz=isoM_mz,
                                                                    lmz=isoMP_mz,
                                                                    tmz=current_Cn * c13c12_offset_charged,
                                                                    xCount=current_Cn,
                                                                    scanIndex=cur_native_scan_index,
                                                                    scanid=cur_scan.id,
                                                                    scantime=cur_scan.retention_time,
                                                                    loading=current_charge,
                                                                    nIntensity=isoM_int,
                                                                    lIntensity=isoMP_int,
                                                                    ionMode=ionMode,
                                                                )
                                                            )

                                                            skip_other_loadings = True
                                                            continue
                                                        elif lowAbundanceIsotopeCutoff and (ratioN is None or ratioL is None):
                                                            cur_signal_detected_partners.append(
                                                                mzFeature(
                                                                    id=len(cur_signal_detected_partners),
                                                                    mz=isoM_mz,
                                                                    lmz=isoMP_mz,
                                                                    tmz=current_Cn * c13c12_offset_charged,
                                                                    xCount=current_Cn,
                                                                    scanIndex=cur_native_scan_index,
                                                                    scanid=cur_scan.id,
                                                                    scantime=cur_scan.retention_time,
                                                                    loading=current_charge,
                                                                    nIntensity=isoM_int,
                                                                    lIntensity=isoMP_int,
                                                                    ionMode=ionMode,
                                                                )
                                                            )

                                                            skip_other_loadings = True
                                                            continue
                                            # endregion

                                        # Isotope pattern validation (useCValidation == 0)
                                        # It is tested, if the expected isotope patterns
                                        # separate for the native and labelled metabolite follow a theoretical pattern
                                        # E.g. native 12C and uniformly / partially 13C-labelled metabolites
                                        # |    <--   Cn   -->    |
                                        # ||                    ||
                                        # |||                  |||
                                        # Necessary mainly for 13C-labelling with mirror-symmetric isotope patterns
                                        # NOTE: - Approach is mainly used for 13C-labelling
                                        if not useDoubleLabelingCombinations and useCIsotopePatternValidation == 0:
                                            # region
                                            # (0.) verify that peak is M and not something else (e.g. M+1, M+1...)
                                            ## TODO improve me. Use seven golden rules or the number of carbon atoms
                                            isoM_m1_index = cur_scan.findMZ(isoM_mz - c13c12_offset_charged, ppm)
                                            isoM_m1_index = cur_scan.getMostIntensePeak(isoM_m1_index[0], isoM_m1_index[1])
                                            if isoM_m1_index != -1:
                                                obs_ratio_Mm1_to_M = cur_scan.intensity_list[isoM_m1_index] / isoM_int
                                                if obs_ratio_Mm1_to_M >= max_other_iso_intensity_ratio:
                                                    continue

                                            # find M+1 peak
                                            isoM_p1_index = cur_scan.findMZ(
                                                isoM_mz + c13c12_offset_charged,
                                                ppm,
                                                start=currentPeakIndex,
                                            )
                                            isoM_p1_index = cur_scan.getMostIntensePeak(isoM_p1_index[0], isoM_p1_index[1])
                                            if isoM_p1_index != -1 or peakCountLeft == 1 or lowAbundanceIsotopeCutoff:
                                                # test certain number of labelled carbon atoms

                                                for current_Cn in sorted(xCounts, reverse=True):
                                                    # stop for impossible carbon atom number

                                                    # find corresponding M' peak (search in labScan, which may be offset)
                                                    isoMP_index = labScan.findMZ(
                                                        isoM_mz + current_Cn * c13c12_offset_charged,
                                                        ppm,
                                                    )
                                                    isoMP_index = labScan.getMostIntensePeak(
                                                        isoMP_index[0],
                                                        isoMP_index[1],
                                                        intensityThres,
                                                    )
                                                    if isoMP_index != -1:
                                                        isoMP_mz = labScan.mz_list[isoMP_index]
                                                        isoMP_int = labScan.intensity_list[isoMP_index]

                                                        # (0.) verify that peak is M' and not something else (e.g. M'-1, M'-2)
                                                        # only for AllExtract experiments
                                                        obs_ratio_MPp1_to_MP = 0
                                                        isoMP_p1_index = labScan.findMZ(
                                                            isoM_mz + (current_Cn + 1) * c13c12_offset_charged,
                                                            ppm,
                                                        )
                                                        isoMP_p1_index = labScan.getMostIntensePeak(
                                                            isoMP_p1_index[0],
                                                            isoMP_p1_index[1],
                                                        )
                                                        if isoMP_p1_index != -1:
                                                            obs_ratio_MPp1_to_MP = labScan.intensity_list[isoMP_p1_index] / isoMP_int

                                                        if not metabolisationExperiment:
                                                            if obs_ratio_MPp1_to_MP >= max_other_iso_intensity_ratio:
                                                                continue

                                                        # (1.) check if M' and M ratio are as expected
                                                        if checkRatio:
                                                            rat = isoM_int / isoMP_int
                                                            if minRatio <= rat <= maxRatio:
                                                                pass  ## ratio check passed
                                                            else:
                                                                continue  ## ratio check not passed

                                                        ## no isotopolog verification needs to be performed
                                                        if peakCountLeft == 1 and peakCountRight == 1:
                                                            cur_signal_detected_partners.append(
                                                                mzFeature(
                                                                    id=next_signal_pair_id,
                                                                    mz=isoM_mz,
                                                                    lmz=isoMP_mz,
                                                                    tmz=isoMP_mz - isoM_mz,
                                                                    xCount=current_Cn,
                                                                    scanIndex=cur_native_scan_index,
                                                                    scanid=cur_scan.id,
                                                                    scantime=cur_scan.retention_time,
                                                                    loading=current_charge,
                                                                    nIntensity=isoM_int,
                                                                    lIntensity=isoMP_int,
                                                                    ionMode=ionMode,
                                                                )
                                                            )
                                                            next_signal_pair_id += 1
                                                            skip_other_loadings = True
                                                            continue

                                                        # find M'-1 peak
                                                        isoMP_m1_index = labScan.findMZ(
                                                            isoM_mz + (current_Cn - 1) * c13c12_offset_charged,
                                                            ppm,
                                                        )
                                                        isoMP_m1_index = labScan.getMostIntensePeak(
                                                            isoMP_m1_index[0],
                                                            isoMP_m1_index[1],
                                                        )
                                                        theoretical_ratio_native = theoretical_ratios_native[current_Cn][1]
                                                        theoretical_ratio_labeled = theoretical_ratios_labeled[current_Cn][1]

                                                        # 2. check if the observed M'-1/M' ratio fits the theoretical one
                                                        if peakCountRight == 1:
                                                            pass
                                                        elif isoMP_m1_index == -1:
                                                            if lowAbundanceIsotopeCutoff and isoMP_int * theoretical_ratio_labeled <= isotopologIntensityThres:
                                                                pass
                                                            else:
                                                                continue
                                                        elif isoMP_m1_index != -1:
                                                            isoMP_m1_int = labScan.intensity_list[isoMP_m1_index]
                                                            observed_ratio = isoMP_m1_int / isoMP_int
                                                            if abs(theoretical_ratio_labeled - observed_ratio) <= intensityErrorL:
                                                                pass
                                                            elif lowAbundanceIsotopeCutoff and isoMP_m1_int <= isotopologIntensityThres:
                                                                pass
                                                            else:
                                                                continue
                                                        else:
                                                            continue

                                                        # 3. check if the observed M+1/M ratio fits the theoretical one (native, always in curScan)
                                                        if peakCountLeft == 1:
                                                            pass
                                                        elif isoM_p1_index == -1:
                                                            if lowAbundanceIsotopeCutoff and isoM_int * (theoretical_ratio_native + obs_ratio_MPp1_to_MP) <= isotopologIntensityThres:
                                                                pass
                                                            else:
                                                                continue
                                                        elif isoM_p1_index != -1:
                                                            isoM_p1_int = cur_scan.intensity_list[isoM_p1_index]
                                                            observed_ratio = isoM_p1_int / isoM_int
                                                            if abs(theoretical_ratio_native + obs_ratio_MPp1_to_MP - observed_ratio) <= intensityErrorN:
                                                                pass
                                                            elif lowAbundanceIsotopeCutoff and isoM_p1_int <= isotopologIntensityThres:
                                                                pass
                                                            else:
                                                                continue
                                                        else:
                                                            continue

                                                        # All verification criteria are passed, store the ion signal pair
                                                        # for further processing
                                                        cur_signal_detected_partners.append(
                                                            mzFeature(
                                                                id=next_signal_pair_id,
                                                                mz=isoM_mz,
                                                                lmz=isoMP_mz,
                                                                tmz=isoMP_mz - isoM_mz,
                                                                xCount=current_Cn,
                                                                scanIndex=cur_native_scan_index,
                                                                scanid=cur_scan.id,
                                                                scantime=cur_scan.retention_time,
                                                                loading=current_charge,
                                                                nIntensity=isoM_int,
                                                                lIntensity=isoMP_int,
                                                                ionMode=ionMode,
                                                            )
                                                        )
                                                        next_signal_pair_id += 1
                                                        skip_other_loadings = True
                                            # endregion

                                        # labeling patterns derived from one or more labeling-elements (e.g. 13C and D)
                                        # Currently, onle the m/z delta is checked but no isotopolog distribution
                                        # E.g. 13CD3
                                        # |  <-- 13CD3 -->  |
                                        # ||                ||
                                        # |||              ||||
                                        # NOTE: currently, the labeling elements must be defined in the code
                                        # NOTE: The option must be activated and the other two options must be deactivated
                                        if useDoubleLabelingCombinations:
                                            # find M+1 peak
                                            isoM_p1_index = cur_scan.findMZ(
                                                isoM_mz + c13c12_offset / current_charge,
                                                ppm,
                                                start=currentPeakIndex,
                                            )
                                            isoM_p1_index = cur_scan.getMostIntensePeak(isoM_p1_index[0], isoM_p1_index[1])
                                            intIsoM_p1 = 0
                                            if isoM_p1_index != -1:
                                                intIsoM_p1 = cur_scan.intensity_list[isoM_p1_index]

                                            isoM_m1_index = cur_scan.findMZ(
                                                isoM_mz - c13c12_offset / current_charge,
                                                ppm,
                                                start=currentPeakIndex,
                                            )
                                            isoM_m1_index = cur_scan.getMostIntensePeak(isoM_m1_index[0], isoM_m1_index[1])
                                            intIsoM_m1 = 0
                                            if isoM_m1_index != -1:
                                                intIsoM_m1 = cur_scan.intensity_list[isoM_m1_index]

                                            if intIsoM_p1 < isoM_int and intIsoM_m1 < isoM_int and (isoM_p1_index != -1 or peakCountLeft == 1 or lowAbundanceIsotopeCutoff):
                                                # test certain number of labelled carbon atoms

                                                for comb in combs:
                                                    # find corresponding M' peak (search in labScan, which may be offset)
                                                    isoMP_index = labScan.findMZ(
                                                        isoM_mz + comb.mzdelta / current_charge,
                                                        ppm,
                                                    )
                                                    isoMP_index = labScan.getMostIntensePeak(
                                                        isoMP_index[0],
                                                        isoMP_index[1],
                                                        intensityThres,
                                                    )
                                                    if isoMP_index != -1:
                                                        isoMP_mz = labScan.mz_list[isoMP_index]
                                                        isoMP_int = labScan.intensity_list[isoMP_index]

                                                        # (1.) check if M' and M ratio are as expected
                                                        if False:
                                                            rat = isoM_int / isoMP_int
                                                            if minRatio <= rat <= maxRatio:
                                                                pass  ## ratio check passed
                                                            else:
                                                                continue  ## ratio check not passed

                                                        # find M'-1 peak
                                                        isoMP_m1_index = labScan._findMZGeneric(
                                                            isoM_mz + (comb.mzdelta - 1.00628 * (1.0 + isoM_mz * ppm / 1000000)) / current_charge,
                                                            isoM_mz + (comb.mzdelta - 1.00335 * (1.0 - isoM_mz * ppm / 1000000)) / current_charge,
                                                        )
                                                        isoMP_m1_index = labScan.getMostIntensePeak(
                                                            isoMP_m1_index[0],
                                                            isoMP_m1_index[1],
                                                        )

                                                        if isoMP_m1_index != -1:
                                                            isoPeakIntensity = labScan.intensity_list[isoMP_m1_index]
                                                            rat = isoPeakIntensity / isoMP_int

                                                            if rat <= 0.75:
                                                                pass
                                                            else:
                                                                continue

                                                        # find M'+1 peak
                                                        isoMP_p1_index = labScan._findMZGeneric(
                                                            isoM_mz + (comb.mzdelta + 1.00335 * (1.0 - isoM_mz * ppm / 1000000)) / current_charge,
                                                            isoM_mz + (comb.mzdelta + 1.00628 * (1.0 + isoM_mz * ppm / 1000000)) / current_charge,
                                                        )
                                                        isoMP_p1_index = labScan.getMostIntensePeak(
                                                            isoMP_p1_index[0],
                                                            isoMP_p1_index[1],
                                                        )

                                                        if isoMP_p1_index != -1:
                                                            isoPeakIntensity = labScan.intensity_list[isoMP_p1_index]
                                                            rat = isoPeakIntensity / isoMP_int

                                                            if rat <= 0.9:
                                                                pass
                                                            else:
                                                                continue

                                                        # All verification criteria are passed, store the ion signal pair
                                                        # for further processing
                                                        cur_signal_detected_partners.append(
                                                            mzFeature(
                                                                id=len(cur_signal_detected_partners),
                                                                mz=isoM_mz,
                                                                lmz=labScan.mz_list[isoMP_index],
                                                                tmz=comb.mzdelta / current_charge,
                                                                xCount=fT.flatToString(comb.atoms),
                                                                scanIndex=cur_native_scan_index,
                                                                scanid=cur_scan.id,
                                                                scantime=cur_scan.retention_time,
                                                                loading=current_charge,
                                                                nIntensity=isoM_int,
                                                                lIntensity=isoMP_int,
                                                                ionMode=ionMode,
                                                            )
                                                        )

                                                        skip_other_loadings = True
                                                        # endregion

                            cur_scan_detected_signal_pairs.extend(cur_signal_detected_partners)

                detected_signal_pairs.extend(cur_scan_detected_signal_pairs)
                cur_native_scan_index += 1

        except Exception:
            print(f"Error in findIsoPairs_matchPartners for scan {cur_scan.id} (index {cur_native_scan_index}):")
            traceback.print_exc()

    return detected_signal_pairs

from __future__ import absolute_import, division, print_function
import base64
import struct
import sys
import xml.parsers.expat
import zlib
from xml.dom.minidom import parse
import numpy as np
import pymzml
from .MSScan import MS1Scan, MS2Scan
from .utils import Bunch


# reads and holds the information of one MZXML file
class Chromatogram:
    def __init__(self):
        self.msLevel = 0
        self.current_tag = ""
        self.tag_level = 0
        self.MS1_list = []
        self.MS2_list = []
        self.msInstruments = {}

    def getMS1ScanCount(self):
        return len(self.MS1_list)

    def getMS1SignalCount(self):
        c = 0
        for scan in self.MS1_list:
            c += len(scan.mz_list)
        return c

    def getAllMS1Intensities(self):
        h = []
        for scan in self.MS1_list:
            h.append(scan.intensity_list)
        return np.concatenate(h) if h else np.array([])

    def getIthMS1Scan(self, index, filterLine):
        i = 0
        for scan in self.MS1_list:
            if scan.filter_line == filterLine:
                if i == index:
                    return scan
                i += 1
        return None

    def getMS1ScanByNum(self, scanNum):
        return self.MS1_list[scanNum]

    def getClosestMS1Scan(self, scanTimeMin, filterLine=""):
        fscan = self.MS1_list[0]
        for scan in self.MS1_list:
            if scan.filter_line == filterLine and abs(scan.retention_time / 60.0 - scanTimeMin) < abs(fscan.retention_time / 60.0 - scanTimeMin):
                fscan = scan
        return fscan

    def getScanByID(self, id):
        for scan in self.MS1_list:
            if scan.id == id:
                return scan
        for scan in self.MS2_list:
            if scan.id == id:
                return scan

    def getFilterLines(
        self,
        includeMS1=True,
        includeMS2=False,
        includePosPolarity=True,
        includeNegPolarity=True,
    ):
        filterLines = set()
        if includeMS1:
            for scan in self.MS1_list:
                if (includePosPolarity and scan.polarity == "+") or (includeNegPolarity and scan.polarity == "-"):
                    filterLines.add(scan.filter_line)
        if includeMS2:
            for scan in self.MS2_list:
                if (includePosPolarity and scan.polarity == "+") or (includeNegPolarity and scan.polarity == "-"):
                    filterLines.add(scan.filter_line)

        return filterLines

    def getFilterLinesExtended(
        self,
        includeMS1=True,
        includeMS2=False,
        includePosPolarity=True,
        includeNegPolarity=True,
    ):
        filterLines = {}
        if includeMS1:
            for scan in self.MS1_list:
                if (includePosPolarity and scan.polarity == "+") or (includeNegPolarity and scan.polarity == "-"):
                    if scan.filter_line not in filterLines.keys():
                        filterLines[scan.filter_line] = Bunch(
                            scanType="MS1",
                            polarity=scan.polarity,
                            targetStartTime=10000000,
                            targetEndTime=0,
                        )
                    filterLines[scan.filter_line].targetStartTime = min(
                        filterLines[scan.filter_line].targetStartTime,
                        scan.retention_time,
                    )
                    filterLines[scan.filter_line].targetEndTime = min(filterLines[scan.filter_line].targetEndTime, scan.retention_time)

        if includeMS2:
            for scan in self.MS2_list:
                if (includePosPolarity and scan.polarity == "+") or (includeNegPolarity and scan.polarity == "-"):
                    if scan.filter_line not in filterLines.keys():
                        filterLines[scan.filter_line] = Bunch(
                            scanType="MS2",
                            polarity=scan.polarity,
                            targetStartTime=10000000,
                            targetEndTime=0,
                            preCursorMz=[],
                            collisionEnergy=0,
                            scanTimes=[],
                        )
                    filterLines[scan.filter_line].targetStartTime = min(
                        filterLines[scan.filter_line].targetStartTime,
                        scan.retention_time,
                    )
                    filterLines[scan.filter_line].targetEndTime = max(filterLines[scan.filter_line].targetEndTime, scan.retention_time)
                    filterLines[scan.filter_line].preCursorMz.append(scan.precursor_mz)
                    filterLines[scan.filter_line].collisionEnergy = scan.collisionEnergy
                    filterLines[scan.filter_line].scanTimes.append(scan.retention_time)

            for k, v in filterLines.items():
                if v.scanType == "MS2":
                    v.preCursorMz = sum(v.preCursorMz) / len(v.preCursorMz)

        return filterLines

    def getFilterLinesPerPolarity(self, includeMS1=True, includeMS2=False):
        filterLines = {"+": set(), "-": set()}
        if includeMS1:
            for scan in self.MS1_list:
                filterLines[scan.polarity].add(scan.filter_line)
        if includeMS2:
            for scan in self.MS2_list:
                filterLines[scan.polarity].add(scan.filter_line)

        return filterLines

    def getPolarities(self):
        polarities = set()
        for scan in self.MS1_list:
            polarities.add(scan.polarity)
        for scan in self.MS2_list:
            polarities.add(scan.polarity)

        return polarities

    def getTIC(self, filterLine="", useMS2=False):
        msLevelList = self.MS2_list if useMS2 else self.MS1_list

        if filterLine == "":
            # Fast path when no filter is needed
            TIC = np.array([scan.total_ion_current for scan in msLevelList], dtype=np.float64)
            times = np.array([scan.retention_time for scan in msLevelList], dtype=np.float64)
            scanIds = np.array([scan.id for scan in msLevelList], dtype=np.int64)
        else:
            # Filtered path
            filtered_scans = [scan for scan in msLevelList if scan.filter_line == filterLine]
            TIC = np.array([scan.total_ion_current for scan in filtered_scans], dtype=np.float64)
            times = np.array([scan.retention_time for scan in filtered_scans], dtype=np.float64)
            scanIds = np.array([scan.id for scan in filtered_scans], dtype=np.int64)

        return TIC, times, scanIds

    # returns a specific area (2-dimensionally bound in rt and mz direction) of the LC-HRMS data
    def getArea(self, startScan, endScan, mz, ppm, filterLine="", intThreshold=0):
        scans = []
        times = []
        scanIDs = []

        scannum = 0
        for scan in self.MS1_list:
            if filterLine == "" or scan.filter_line == filterLine:
                if endScan >= scannum >= startScan:
                    bounds = scan.findMZ(mz, ppm)
                    curScan = []
                    if bounds[0] != -1:
                        mz_subset = scan.mz_list[bounds[0] : bounds[1] + 1]
                        intensity_subset = scan.intensity_list[bounds[0] : bounds[1] + 1]
                        mask = intensity_subset >= intThreshold
                        curScan = list(zip(mz_subset[mask], intensity_subset[mask]))
                    scans.append(curScan)
                    times.append(scan.retention_time)
                    scanIDs.append(scan.id)

                scannum += 1

        return scans, np.array(times), np.array(scanIDs)

    # returns a specific area (2-dimensionally bound in rt and mz direction) of the LC-HRMS data
    def getSpecificArea(self, startTime, endTime, mzmin, mzmax, filterLine="", intThreshold=0):
        scans = []
        times = []
        scanIDs = []

        scannum = 0
        for scan in self.MS1_list:
            if filterLine == "" or scan.filter_line == filterLine:
                if startTime <= scan.retention_time <= endTime:
                    bounds = scan._findMZGeneric(mzmin, mzmax)
                    curScan = []
                    if bounds[0] != -1:
                        mz_subset = scan.mz_list[bounds[0] : bounds[1] + 1]
                        intensity_subset = scan.intensity_list[bounds[0] : bounds[1] + 1]
                        mask = intensity_subset >= intThreshold
                        curScan = list(zip(mz_subset[mask], intensity_subset[mask]))
                    scans.append(curScan)
                    times.append(scan.retention_time)
                    scanIDs.append(scan.id)

                scannum += 1

        return scans, np.array(times), np.array(scanIDs)

    # returns an eic. Single MS peaks and such below a certain threshold may be removed
    # Returns: eic (2D array [2, n] where eic[0,:] = times, eic[1,:] = intensities),
    #          times (1D array), scanIds (1D array), mzs (1D array)
    def getEIC(
        self,
        mz,
        ppm,
        filterLine="",
        removeSingles=True,
        intThreshold=0,
        useMS1=True,
        useMS2=False,
        startTime=0,
        endTime=1000000,
    ):
        eic = []
        times = []
        scanIds = []
        mzs = []

        if useMS1:
            for scan in self.MS1_list:
                if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                    bounds = scan.findMZ(mz, ppm)
                    if bounds[0] != -1:
                        intensity_subset = scan.intensity_list[bounds[0] : (bounds[1] + 1)]
                        uIndex = np.argmax(intensity_subset)
                        eic.append(scan.intensity_list[bounds[0] + uIndex])
                        mzs.append(scan.mz_list[bounds[0] + uIndex])
                    else:
                        eic.append(0)
                        mzs.append(-1)
                    times.append(scan.retention_time)
                    scanIds.append(scan.id)

        if useMS2:
            for scan in self.MS2_list:
                if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                    bounds = scan.findMZ(mz, ppm)
                    if bounds[0] != -1:
                        intensity_subset = scan.intensity_list[bounds[0] : (bounds[1] + 1)]
                        uIndex = np.argmax(intensity_subset)
                        eic.append(scan.intensity_list[bounds[0] + uIndex])
                        mzs.append(scan.mz_list[bounds[0] + uIndex])
                    else:
                        eic.append(0)
                        mzs.append(-1)
                    times.append(scan.retention_time)
                    scanIds.append(scan.id)

        # Convert to numpy arrays
        eic = np.array(eic, dtype=np.float64)
        times = np.array(times, dtype=np.float64)
        scanIds = np.array(scanIds, dtype=np.int64)
        mzs = np.array(mzs, dtype=np.float64)

        # Apply threshold filter
        mask = eic < intThreshold
        if len(mask) > 0:
            mask[0] = False  # Don't filter first element
        eic[mask] = 0
        mzs[mask] = -1

        # Remove singles if requested
        if removeSingles and len(eic) > 2:
            for i in range(1, len(eic) - 1):
                if eic[i - 1] == 0 and eic[i + 1] == 0:
                    eic[i] = 0
                    mzs[i] = -1

        return eic, times, scanIds, mzs

    def getSignalCount(
        self,
        filterLine="",
        removeSingles=True,
        intThreshold=0,
        useMS1=True,
        useMS2=False,
        startTime=0,
        endTime=1000000,
    ):
        totSignals = 0
        for scan in self.MS1_list + self.MS2_list:
            if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                totSignals += np.sum(scan.intensity_list >= intThreshold)

        return totSignals

    def getMinMaxAvgSignalIntensities(
        self,
        filterLine="",
        removeSingles=True,
        intThreshold=0,
        useMS1=True,
        useMS2=False,
        startTime=0,
        endTime=1000000,
    ):
        minInt = np.inf
        maxInt = 0
        avgsum = 0
        avgcount = 0
        for scan in self.MS1_list + self.MS2_list:
            if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                mask = scan.intensity_list >= intThreshold
                uVals = scan.intensity_list[mask]
                if len(uVals) == 0:
                    uVals = np.array([0])

                minInt = np.minimum(minInt, np.min(uVals))
                maxInt = np.maximum(maxInt, np.max(uVals))
                avgsum += np.sum(uVals)
                avgcount += len(uVals)

        if avgcount == 0:
            return 0, 0, 0
        else:
            return minInt, maxInt, avgsum / avgcount

    # converts base64 coded spectrum in an mz and intensity array
    def decode_spectrum(self, line, compression=None, precision=32):
        mz_list = []
        intensity_list = []

        if line is not None and len(line) > 0:
            decoded = base64.decodebytes(line.encode("utf-8"))
            if compression == "zlib":
                decoded = zlib.decompress(decoded)

            if precision == 64:
                tmp_size = len(decoded) // 8
                unpack_format1 = ">%dd" % tmp_size
            else:
                tmp_size = len(decoded) // 4
                unpack_format1 = ">%df" % tmp_size

            unpacked = struct.unpack(unpack_format1, decoded)
            # Use numpy array slicing for efficiency
            data = np.array(unpacked, dtype=np.float64)
            mz_list = data[0::2]  # Every even index
            intensity_list = data[1::2]  # Every odd index

        return mz_list, intensity_list

    # xml parser - start element handler
    def _start_element(self, name, attrs):
        self.tag_level += 1
        self.current_tag = name
        # print("start tag", name)

        if name == "parentFile" and "fileName" in attrs:
            self.parentFile = attrs["fileName"]

        if name == "msInstrument" and "msInstrumentID" in attrs:
            self.msInstruments[attrs["msInstrumentID"]] = {}
            self.lastMSInstrument = self.msInstruments[attrs["msInstrumentID"]]

        if name == "msModel" and len(self.msInstruments) > 0:
            self.lastMSInstrument["msModel"] = attrs["value"]

        if name == "msMassAnalyzer" and len(self.msInstruments) > 0:
            self.lastMSInstrument["msMassAnalyzer"] = attrs["value"]

        if name == "precursorMz":
            self.MS2_list[-1].precursor_intensity = float(attrs["precursorIntensity"])
            self.MS2_list[-1].precursor_charge = 0
            if "precursorCharge" in attrs:
                self.MS2_list[-1].precursor_charge = int(attrs["precursorCharge"])
            if "activationMethod" in attrs:
                self.MS2_list[-1].activationMethod = str(attrs["activationMethod"])

        # Extract cvParams from scan elements (for mzML format)
        if name == "cvParam" and len(self.MS2_list) > 0 and self.msLevel == 2:
            cv_dict = {"accession": attrs.get("accession", ""), "cvRef": attrs.get("cvRef", ""), "name": attrs.get("name", ""), "value": attrs.get("value", "")}
            self.MS2_list[-1].cvParams.append(cv_dict)
        elif name == "cvParam" and len(self.MS1_list) > 0 and self.msLevel == 1:
            cv_dict = {"accession": attrs.get("accession", ""), "cvRef": attrs.get("cvRef", ""), "name": attrs.get("name", ""), "value": attrs.get("value", "")}
            self.MS1_list[-1].cvParams.append(cv_dict)

        if name == "scan":
            self.curScan = self.curScan + 1

            self.msLevel = int(attrs["msLevel"])
            if self.msLevel == 1:
                tmp_ms = MS1Scan()
            elif self.msLevel == 2:
                tmp_ms = MS2Scan()
            else:
                print("What is it?", attrs)
                sys.exit(1)

            tmp_ms.id = int(attrs["num"])

            tmp_ms.peak_count = int(attrs["peaksCount"])
            tmp_ms.peak_count_tag = int(attrs["peaksCount"])
            tmp_ms.retention_time = float(attrs["retentionTime"].strip("PTS"))
            if tmp_ms.peak_count > 0:
                if "lowMz" in attrs:
                    tmp_ms.low_mz = float(attrs["lowMz"])
                if "highMz" in attrs:
                    tmp_ms.high_mz = float(attrs["highMz"])
                if "basePeakMz" in attrs:
                    tmp_ms.base_peak_mz = float(attrs["basePeakMz"])
                if "basePeakIntensity" in attrs:
                    tmp_ms.base_peak_intensity = float(attrs["basePeakIntensity"])
            if "totIonCurrent" in attrs:
                tmp_ms.total_ion_current = float(attrs["totIonCurrent"])
            else:
                tmp_ms.total_ion_current = 0
            tmp_ms.list_size = 0
            tmp_ms.polarity = str(attrs["polarity"])
            tmp_ms.encoded_mz = ""
            tmp_ms.encoded_intensity = ""
            tmp_ms.encodedData = ""
            tmp_ms.mz_list = []
            tmp_ms.intensity_list = []
            tmp_ms.msInstrumentID = ""
            if "msInstrumentID" in attrs:
                tmp_ms.msInstrumentID = attrs["msInstrumentID"]
            tmp_ms.filter_line = "N/A"
            if "filterLine" in attrs:
                tmp_ms.filter_line = attrs["filterLine"] + " (pol: %s)" % tmp_ms.polarity
            elif len(self.msInstruments) == 1:
                tmp_ms.filter_line = "%s (MS lvl: %d, pol: %s)" % (
                    self.msInstruments["1"]["msModel"],
                    self.msLevel,
                    tmp_ms.polarity,
                )
            else:
                tmp_ms.filter_line = "%s (MS lvl: %d, pol: %s)" % (
                    "Unknown",
                    self.msLevel,
                    tmp_ms.polarity,
                )

            if self.msLevel == 1:
                self.MS1_list.append(tmp_ms)
            elif self.msLevel == 2:
                if "collisionEnergy" in attrs:
                    tmp_ms.collisionEnergy = float(attrs["collisionEnergy"])
                else:
                    tmp_ms.collisionEnergy = -1
                tmp_ms.ms1_id = self.MS1_list[-1].id
                self.MS2_list.append(tmp_ms)

        if name == "peaks":
            if self.msLevel == 1:
                curScan = self.MS1_list[-1]
            elif self.msLevel == 2:
                curScan = self.MS2_list[-1]

            curScan.compression = None
            if "compressionType" in attrs:
                curScan.compression = str(attrs["compressionType"])
            curScan.precision = int(attrs["precision"])

    def _findMZGeneric(self, mzleft, mzright, mzlist):
        if len(mzlist) == 0:
            return -1, -1

        min = 0
        max = len(mzlist) - 1
        peakCount = len(mzlist)

        while min <= max:
            cur = int((max + min) // 2)

            if mzleft <= mzlist[cur] <= mzright:
                leftBound = cur
                while leftBound > 0 and mzlist[leftBound - 1] >= mzleft:
                    leftBound -= 1

                rightBound = cur
                while (rightBound + 1) < peakCount and mzlist[rightBound + 1] <= mzright:
                    rightBound += 1

                return leftBound, rightBound

            if mzlist[cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1, -1

    # xml parser - end element handler
    def _end_element(self, name):
        # print("end tag", name)

        if name == "scan":
            if self.msLevel == 1:
                curScan = self.MS1_list[-1]
            elif self.msLevel == 2:
                curScan = self.MS2_list[-1]

            mz_list, intensity_list = self.decode_spectrum(
                curScan.encodedData,
                compression=curScan.compression,
                precision=curScan.precision,
            )

            ## Remove peaks below threshold
            if self.intensityCutoff > 0:
                mask = (intensity_list > self.intensityCutoff) & (mz_list > 0)
                mz_list = mz_list[mask]
                intensity_list = intensity_list[mask]

            ## Remove peaks with unwanted mz values
            if self.mzFilter is not None:
                keep_mask = np.zeros(len(mz_list), dtype=bool)
                for ps, pe in self.mzFilter:
                    a, b = self._findMZGeneric(ps, pe, mz_list)
                    if a != -1:
                        keep_mask[a : b + 1] = True
                mz_list = mz_list[keep_mask]
                intensity_list = intensity_list[keep_mask]

            assert len(mz_list) == len(intensity_list)

            curScan.mz_list = mz_list
            curScan.intensity_list = intensity_list
            curScan.peak_count = len(mz_list)
            curScan.encodedData = None

        if name == "precursorMz":
            self.MS2_list[-1].precursor_mz = float(self.MS2_list[-1].precursor_mz_data)
            # self.MS2_list[-1].filter_line="%s (PreMZ: %.2f ColEn: %.1f ActMet: %s)"%(self.MS2_list[-1].filter_line, self.MS2_list[-1].precursor_mz, self.MS2_list[-1].collisionEnergy, self.MS2_list[-1].activationMethod if hasattr(self.MS2_list[-1], "activationMethod") else "-")

        self.tag_level -= 1
        self.current_tag = ""
        self.msLevel == 0

    # xml parser - CDATA handler
    def _char_data(self, data):
        if self.current_tag == "precursorMz":
            self.MS2_list[-1].precursor_mz_data += data

        if self.current_tag == "peaks" and not self.ignorePeaksData:
            if self.msLevel == 1:
                self.MS1_list[-1].encodedData += data

            elif self.msLevel == 2:
                self.MS2_list[-1].encodedData += data

    # parses an mzxml file and stores its information in the respective object.
    # If intensityCutoff is set, all MS peaks below this threshold will be discarded.
    # If ignoreCharacterData is set, only the metainformation will be parsed but not the actual MS data
    def parseMZXMLFile(self, filename_xml, intensityCutoff=-1, ignoreCharacterData=False):
        self.intensityCutoff = intensityCutoff
        self.curScan = 1

        expat = xml.parsers.expat.ParserCreate()
        expat.StartElementHandler = self._start_element
        expat.EndElementHandler = self._end_element
        self.ignorePeaksData = ignoreCharacterData
        expat.CharacterDataHandler = self._char_data

        ##expat.Parse(content_listd)
        expat.ParseFile(open(filename_xml, "rb"))

        if not ignoreCharacterData:
            for scan in self.MS1_list:
                assert len(scan.mz_list) == len(scan.intensity_list) == scan.peak_count
                # if intensityCutoff<0:
                #    assert scan.peak_count == scan.peak_count_tag

        signals = 0
        for msscan in self.MS1_list:
            signals += len(msscan.mz_list)

        for ionMode in ["+", "-"]:
            precursors = []
            for msmsscan in self.MS2_list:
                if msmsscan.polarity == ionMode:
                    precursors.append(msmsscan.precursor_mz)

            precursors = sorted(precursors)

            packets = []
            curpack = []

            for precursorMZ in precursors:
                if len(curpack) == 0:
                    curpack.append(precursorMZ)
                else:
                    mzmean = sum(curpack) / len(curpack)
                    if abs(mzmean - precursorMZ) / precursorMZ * 1e6 > 10:
                        packets.append(curpack)
                        curpack = []
                    curpack.append(precursorMZ)
            if len(curpack) > 0:
                packets.append(curpack)

            for curpack in packets:
                meanmz = sum(curpack) / len(curpack)
                # print(ionMode, meanmz, (meanmz-min(curpack))/meanmz*1E6, (max(curpack)-meanmz)/meanmz*1E6, curpack)

            for msmsscan in self.MS2_list:
                if msmsscan.polarity == ionMode:
                    for curpack in packets:
                        meanmz = sum(curpack) / len(curpack)
                        if abs(msmsscan.precursor_mz - meanmz) / msmsscan.precursor_mz * 1e6 <= 10:
                            # msmsscan.precursor_mz=meanmz
                            msmsscan.filter_line = "%s (PreMZ: %.4f [%.6f-%6f; %.1f ppm] ColEn: %.1f ActMet: %s)" % (
                                msmsscan.filter_line,
                                meanmz,
                                min(curpack),
                                max(curpack),
                                (max(curpack) - min(curpack)) / meanmz * 1e6,
                                msmsscan.collisionEnergy,
                                msmsscan.activationMethod if hasattr(msmsscan, "activationMethod") else "-",
                            )

    def parseMzMLFile(self, filename_xml, intensityCutoff, ignoreCharacterData):
        run = pymzml.run.Reader(filename_xml)

        for spectrum in run:
            if "time array" in spectrum or spectrum["id"] is None:
                continue

            try:
                msLevel = int(spectrum["ms level"])
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error: Cannot determine MS level for spectrum {spectrum['id']}: {e}")
                continue

            if msLevel == 1:
                tmp_ms = MS1Scan()
            elif msLevel == 2:
                tmp_ms = MS2Scan()
            else:
                print("What is it?", msLevel, spectrum["id"])
                sys.exit(1)

            if "positive scan" in spectrum:
                tmp_ms.polarity = "+"
            elif "negative scan" in spectrum:
                tmp_ms.polarity = "-"
            else:
                raise RuntimeError("No polarity for scan available")

            tmp_ms.id = int(spectrum["id"])
            if msLevel == 1:
                if "filter string" in spectrum.__dict__.keys():
                    tmp_ms.filter_line = spectrum["filter string"]
                else:
                    tmp_ms.filter_line = f"NA // MSLevel: {msLevel}, polarity: {tmp_ms.polarity}"
            if msLevel == 2:
                filter_str = None
                try:
                    if "filter string" in spectrum:
                        filter_str = spectrum["filter string"]
                except (KeyError, TypeError):
                    filter_str = None
                if filter_str:
                    tmp_ms.filter_line = filter_str
                    tmp_ms.filter_string = filter_str
                else:
                    tmp_ms.filter_line = f"NA // MSLevel: {msLevel}, polarity: {tmp_ms.polarity}"

            tmp_ms.peak_count = len(spectrum.peaks(peak_type="centroided"))
            tmp_ms.retention_time = spectrum.scan_time_in_minutes() * 60.0

            # if tmp_ms.peak_count > 0:
            tmp_ms.total_ion_current = spectrum["total ion current"]
            tmp_ms.list_size = 0
            # Keep as numpy arrays directly from pymzml
            tmp_ms.mz_list = spectrum.peaks(peak_type="centroided")[:, 0].copy()
            tmp_ms.intensity_list = spectrum.peaks(peak_type="centroided")[:, 1].copy()
            tmp_ms.msInstrumentID = ""

            if msLevel == 1:
                self.MS1_list.append(tmp_ms)
            elif msLevel == 2:
                tmp_ms.precursor_mz = spectrum.selected_precursors[0].get("mz", 0)
                tmp_ms.precursor_intensity = spectrum.selected_precursors[0].get("i", 0)
                tmp_ms.precursor_charge = spectrum.selected_precursors[0].get("charge", 0)
                self.MS2_list.append(tmp_ms)

        precursors = []
        i = 0
        polarities = set()
        for ms2Scan in self.MS2_list:
            precursors.append(Bunch(precursormz=ms2Scan.precursor_mz, id=i, polarity=ms2Scan.polarity))
            polarities.add(ms2Scan.polarity)
            i = i + 1

        from .utils import mean

        for polarity in polarities:
            precursorsTemp = [pc for pc in precursors if pc.polarity == polarity]
            precursorsTemp = sorted(precursorsTemp, key=lambda x: x.precursormz)

            lastmz = -1000
            lastmzs = []

            for i in range(len(precursorsTemp)):
                if (precursorsTemp[i].precursormz - lastmz) >= 25 * lastmz / 1000000.0:
                    if len(lastmzs) > 0:
                        lastmzs[0].precursormz
                        lastmzs[-1].precursormz
                        # print(polarity, minMZ, maxMZ, (maxMZ-minMZ)*1000000./minMZ, "  \n --> ", (precursorsTemp[i].precursormz-lastmz)*1000000./lastmz, ": ", precursorsTemp[i].precursormz)

                        for j in range(len(lastmzs)):
                            self.MS2_list[lastmzs[j].id].filter_line = "MSn %s mean mz precursor: %.5f" % (polarity, mean([pc.precursormz for pc in lastmzs]))
                            self.MS2_list[lastmzs[j].id].precursor_mz = float(mean([pc.precursormz for pc in lastmzs]))
                    lastmz = precursorsTemp[i].precursormz
                    lastmzs = [precursorsTemp[i]]
                else:
                    lastmzs.append(precursorsTemp[i])
            if len(lastmzs) > 0:
                for j in range(len(lastmzs)):
                    self.MS2_list[lastmzs[j].id].filter_line = "MSn %s mean mz precursor: %.5f" % (polarity, mean([pc.precursormz for pc in lastmzs]))
                    self.MS2_list[lastmzs[j].id].precursor_mz = float(mean([pc.precursormz for pc in lastmzs]))

    def parse_file(self, filename_xml, intensityCutoff=-1, ignoreCharacterData=False, mzFilter=None):
        self.mzFilter = mzFilter
        if filename_xml.lower().endswith(".mzxml"):
            return self.parseMZXMLFile(filename_xml, intensityCutoff, ignoreCharacterData)
        elif filename_xml.lower().endswith(".mzml"):
            return self.parseMzMLFile(filename_xml, intensityCutoff, ignoreCharacterData)
        else:
            raise RuntimeError("Invalid file type")

    # updates the data of this MS scan
    # WARNING: unstable, horridly implemented method. SUBJECT OF CHANGE
    def updateScan(self, scan, data):
        scanID = int(scan.getAttribute("num"))

        #  tmp_ms.retention_time = float(attrs['retentionTime'].strip('PTS'))
        if scanID in data and hasattr(data[scanID], "rt"):
            scan.setAttribute("retentionTime", "PT%.3fS" % (data[scanID].rt))

        peaks = None
        for kid in scan.childNodes:
            if kid.nodeName == "peaks":
                peaks = kid
        assert peaks is not None and len(peaks.childNodes) == 1

        if scanID in data and len(data[scanID].mzs) > 0:
            h = data[scanID]
            assert len(h.mzs) == len(h.ints)
            # Convert numpy arrays to bytes for encoding
            mzs = np.asarray(h.mzs, dtype=np.float32)
            ints = np.asarray(h.ints, dtype=np.float32)
            interleaved = np.empty(len(mzs) * 2, dtype=np.float32)
            interleaved[0::2] = mzs
            interleaved[1::2] = ints
            peaks.childNodes[0].nodeValue = base64.encodebytes(interleaved.tobytes(">f")).decode("utf-8").strip().replace("\n", "")
            scan.setAttribute("peaksCount", str(len(h.mzs)))
            scan.setAttribute("lowMz", str(np.min(mzs)))
            scan.setAttribute("highMz", str(np.max(mzs)))
            scan.setAttribute("basePeakMz", "0")
            scan.setAttribute("basePeakIntensity", "0")
            scan.setAttribute("totIonCurrent", str(np.sum(ints)))
        else:
            peaks.childNodes[0].nodeValue = ""
            scan.setAttribute("peaksCount", "0")
            scan.setAttribute("lowMz", "0")
            scan.setAttribute("highMz", "0")
            scan.setAttribute("basePeakMz", "0")
            scan.setAttribute("basePeakIntensity", "0")
            scan.setAttribute("totIonCurrent", "0")

    # updates the data of this mzxml file
    # WARNING: unstable, horridly implemented method. SUBJECT OF CHANGE
    def updateData(self, node, data):
        for kid in node.childNodes:
            if kid.nodeName != "#text":
                if kid.nodeName == "scan":
                    self.updateScan(kid, data)
                else:
                    self.updateData(kid, data)

    # updates the data of this mzxml file
    # WARNING: unstable, horridly implemented method. SUBJECT OF CHANGE
    def resetMZData(self, mzxmlfile, toFile, data):
        dom = parse(mzxmlfile)
        self.updateData(dom, data)
        dom.writexml(open(toFile, "w"))

    def freeMe(self):
        for scan in self.MS2_list:
            scan.freeMe()
        for scan in self.MS1_list:
            scan.freeMe()


if __name__ == "__main__":
    f = "F:/MaxPlanck_FrederikDethloff/exp4/POS/pos-MF-1_25_01_4001.mzXML"
    f = "E:/___Backup/Publications/Manuscripts/MetExtractII/_assets/geoRge/MTBLS213_20170613_073311/CELL_Glc13_05mM_Normo_01.mzXML"
    f = "H:/20181016_472_Alternaria_ExperimentForDFGProposal/FTICRMS/F_negNS300147444mL1zu109_000001.mzXML"

    t = Chromatogram()
    t.parse_file(f)

    for i in t.getFilterLinesPerPolarity(includeMS1=False, includeMS2=True):
        print(i)
        for j in i:
            print("   ", j)

    print(t.getPolarities())

    x = Chromatogram()
    x.parse_file(f)

    scan = x.getMS1ScanByNum(0)
    print(scan.retention_time / 60)

    for mz, inte in zip(scan.mz_list, scan.intensity_list):
        print(mz, inte)

    import matplotlib.pyplot as plt

    plt.vlines(x=scan.mz_list, ymin=0, ymax=scan.intensity_list)

    plt.show()

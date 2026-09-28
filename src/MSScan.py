import numpy as np


# class for one MS scan
class MSScan:
    def __init__(self):
        self.id = 0
        self.peak_count = 0
        self.filter_line = ""
        self.filter_string = ""
        self.retention_time = 0.0
        self.low_mz = 0.0
        self.high_mz = 0.0
        self.polarity = ""
        self.base_peak_mz = 0.0
        self.base_peak_intensity = 0.0
        self.total_ion_current = 0.0
        self.list_size = 0
        self.encoded_mz = ""
        self.encoded_intensity = ""
        self.encodedData = ""
        self.mz_list = np.array([], dtype=np.float64)
        self.intensity_list = np.array([], dtype=np.float64)
        self.msInstrumentID = ""

    @property
    def spectrum(self):
        """Returns spectrum as 2D array: spectrum[0,:] = mz values, spectrum[1,:] = intensities"""
        return np.vstack([self.mz_list, self.intensity_list])

    @spectrum.setter
    def spectrum(self, value):
        """Sets spectrum from 2D array where value[0,:] = mz values, value[1,:] = intensities"""
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 2:
            self.mz_list = value[0, :]
            self.intensity_list = value[1, :]
            self.peak_count = len(self.mz_list)
        else:
            raise ValueError("Spectrum must be a 2D numpy array with shape (2, n)")

    # returns the most abundant ms peak in a given range
    def getMostIntensePeak(self, leftInd, rightInd, minInt=0):
        i = -1
        if leftInd != -1 and rightInd != -1:
            subset = self.intensity_list[leftInd : rightInd + 1]
            mask = subset > minInt
            if np.any(mask):
                local_max_idx = np.argmax(subset[mask])
                # Get the actual index in the masked array
                masked_indices = np.where(mask)[0]
                i = leftInd + masked_indices[local_max_idx]
                self.intensity_list[i]
        return i

    # HELPER METHOD: tests (and returns) if a given mz value is present in the current ms scan
    # binary search is used for efficiency
    def _findMZGeneric(self, mzleft, mzright, start=None, stop=None):
        if len(self.mz_list) == 0:
            return -1, -1

        min = 0 if start is None else start
        max = len(self.mz_list) - 1 if stop is None else stop

        mzlist = self.mz_list
        peakCount = self.peak_count

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

    # tests (and returns) if a given mz value is present within a given error (in ppm) in the current ms scan
    # start and stop are used for iterative purposes and define starting conditions for the following
    # binary search
    def findMZ(self, mz, ppm, start=None, stop=None):
        mzleft = mz * (1.0 - ppm / 1000000.0)
        mzright = mz * (1.0 + ppm / 1000000.0)

        return self._findMZGeneric(mzleft, mzright, start=start, stop=stop)

    def freeMe(self):
        self.intensity_list = np.array([], dtype=np.float64)
        self.mz_list = np.array([], dtype=np.float64)


class MS1Scan(MSScan):
    def __init__(self):
        super().__init__()
        self.cvParams = []  # List of cvParam dicts with accession, cvRef, name, value


class MS2Scan(MSScan):
    def __init__(self):
        super().__init__()
        self.ms1_id = 0
        self.precursor_mz = 0.0
        self.precursor_mz_data = ""
        self.precursor_intensity = 0.0
        self.precursorCharge = 0
        self.collisionEnergy = 0.0
        self.activationMethod = ""
        self.cvParams = []  # List of cvParam dicts with accession, cvRef, name, value

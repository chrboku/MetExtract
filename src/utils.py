import os
import platform
from math import factorial, sqrt
from operator import itemgetter
import numpy
import numpy as np
from scipy.ndimage import gaussian_filter1d
import importlib
import sys
import threading
from multiprocessing import Process, Queue
from sys import exc_info
from traceback import format_tb
import json
import re
import hashlib
from xlrd import open_workbook
from . import PolarsDB
import contextlib
import logging


# get the operating system
def get_app_folder():
    local_folder = "."
    os_name = platform.system()
    if os_name == "Windows":
        local_folder = os.environ["LOCALAPPDATA"]
    elif os_name == "Linux":
        local_folder = os.environ["HOME"]
    elif os_name == "Darwin":
        local_folder = os.environ["HOME"]
    local_folder = os.path.join(local_folder, "MetExtractII")
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    return local_folder


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    y = np.array(y)
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def __smooth(x, window_len=11, window="bartlett"):
    """
    (c) http://www.scipy.org/Cookbook/SignalSmooth (last accessed: 30. May 2012)

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len // 2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Window is non of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = numpy.ones(window_len, "d")
    else:
        w = eval("numpy." + window + "(window_len)")

    y = numpy.convolve(w / w.sum(), s, mode="valid")
    return y


def _smooth(x, window_len=11, window="hanning"):
    x_arr = np.asarray(x, dtype=np.float64)
    x = __smooth(x_arr, window_len, window)
    x = x[int(window_len // 2) : int((len(x) - window_len / 2 + (1 if not (window_len % 2) else 0)))]
    return x


def _smoothTriangle(data, degree, dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""
    data = np.asarray(data, dtype=np.float64)
    triangle = np.concatenate([np.arange(degree), [degree], np.arange(degree)[::-1]]) + 1
    smoothed = []
    triangle_sum = np.sum(triangle)
    for i in range(degree, len(data) - degree * 2):
        point = data[i : i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / triangle_sum)
    if dropVals:
        return np.array(smoothed)
    # Pad with edge values
    pad_len = degree + degree // 2
    smoothed = np.pad(smoothed, (pad_len, 0), mode="edge")
    if len(smoothed) < len(data):
        smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode="edge")
    return smoothed


# smooth data series
def smoothDataSeries(x, y, windowLen=2, polynom=3, window="gaussian", removeNegIntensities=True):
    if window is None or window.lower() == "none":
        return y

    window = window.lower()

    addD = int(round(windowLen / 2.0))

    # Convert to numpy array if not already
    y_arr = np.asarray(y, dtype=np.float64)

    # Pad the array by repeating first and last values
    yi = np.pad(y_arr, (addD, addD), mode="edge")

    if window == "triangle":
        ys = _smoothTriangle(yi, windowLen)
        ys = ys[0 : len(y)]

    elif window == "gaussian":
        ys = gaussian_filter1d(yi, windowLen)
        ys = ys[addD : (len(y) + addD)]

    elif window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        ys = _smooth(yi, window_len=windowLen, window=window)
        ys = ys[addD : (len(y) + addD)]
    elif window == "savitzkygolay":
        ys = savitzky_golay(yi, windowLen, order=polynom)
        ys = ys[addD : (len(y) + addD)]
    else:
        raise Exception("Unsupported smoothing window")

    if removeNegIntensities:
        ys = np.maximum(ys, 0)

    return ys


# HELPER METHOD
# taken from http://www.py2exe.org/index.cgi/HowToDetermineIfRunningFromExe


def main_is_frozen():
    return (
        hasattr(sys, "frozen")  # new py2exe
        or hasattr(sys, "importers")  # old py2exe
        or importlib.util.find_spec("frozen") is not None
    )  # tools/freeze


# method to determine location of python script / executable (not working directory)
def get_main_dir():
    path = ""
    if main_is_frozen():
        path = os.path.dirname(sys.executable).replace("\\", "/")
    else:
        path = os.path.abspath(__file__).replace("\\", "/")[: os.path.abspath(__file__).replace("\\", "/").rfind("/")]

    path = os.path.abspath(path)
    # if the last directory is named "src", return its parent directory
    if os.path.basename(path) == "src":
        path = os.path.dirname(path)

    return path.replace("\\", "/")


class FuncThread(threading.Thread):
    def __init__(self, _target, *args, **kwds):
        threading.Thread.__init__(self)

        self._target = _target
        self._args = args
        self._kwds = kwds

        self._stop = threading.Event()

    def addArg(self, val):
        self._args.append(val)

    def addKwd(self, param, val):
        self._kwds[param] = val

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        self._target(*self._args, **self._kwds)


if __name__ == "__main__" and False:
    # Example usage
    def someOtherFunc(data, key):
        print("someOtherFunc was called : data=%s; key=%s" % (str(data), str(key)))

    t1 = FuncThread(_target=someOtherFunc, data=[1, 2], key=6)
    t1.start()
    t1.join()


def process_func(q, _target, *args, **kwds):
    try:
        ret = _target(*args, **kwds)
    except Exception:
        ex_type, ex_value, tb = exc_info()
        error = ex_type, ex_value, "".join(format_tb(tb))
        ret = None
    else:
        error = None

    q.put((ret, error))


class FuncProcess:
    def __init__(self, _target, *args, **kwds):
        self._target = _target
        self._args = args
        self._kwds = kwds

        self.q = None

    def getQueue(self):
        if self.q is None:
            self.q = Queue()
        return self.q

    def addArg(self, val):
        self._args.append(val)

    def addKwd(self, param, val):
        self._kwds[param] = val

    def start(self):
        self.p = Process(
            target=process_func,
            args=[self.q, self._target] + list(self._args),
            kwargs=self._kwds,
        )
        self.p.start()

    def isAlive(self):
        if self.p is None:
            return False
        return self.p.is_alive()

    def stopped(self):
        return not self.isAlive()

    def join(self):
        if self.p is None:
            return
        self.p.join()

    def terminate(self):
        self.p.terminate()

    def getReturn(self):
        self.p.join()
        ret, error = self.q.get()

        if error:
            ex_type, ex_value, tb_str = error
            message = "%s (in subprocess)\n%s" % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret


def someOtherFunc(data, key):
    print("someOtherFunc was called in separate process (%s): data=%s; key=%s" % (os.getpid(), str(data), str(key)))
    return "someReturn"


if __name__ == "__main__" and False:
    # Example usage
    t1 = FuncProcess(_target=someOtherFunc, data=[1, 2], key=7)
    t1.start()
    print("main process", os.getpid(), "t1 is alive:", t1.isAlive())

    print("t1 is alive:", t1.isAlive())
    t1.join()
    print("t1 is alive:", t1.isAlive())
    print(t1.getReturn())
    t1.terminate()


class CallBackMethod:
    def __init__(self, _target, *args, **kwds):
        self._target = _target
        self._args = args
        self._kwds = kwds

    def run(self, *args, **kwds):
        ar = []
        ar.extend(self._args)
        ar.extend(args)
        kw = {}
        kw.update(self._kwds)
        kw.update(kwds)
        return self._target(*ar, **kw)

    def getRunMethod(self):
        return self.run


if False:

    def callbackMethod(a="hello", b="world"):
        print(a, b)

    cm = CallBackMethod(_target=callbackMethod, b="Joe").getRunMethod()
    cm()


# GENERIC CLASS that is composed of member variables supplied during object construction.
# It implements a generic string representation
class Bunch:
    def __init__(self, _addFollowing=None, _addFollowingDefaultValue=None, **kwds):
        if _addFollowing is not None:
            if isinstance(_addFollowing, list):
                for att in _addFollowing:
                    setattr(self, att, _addFollowingDefaultValue)
            elif isinstance(_addFollowing, dict):
                for att, val in _addFollowing.items():
                    setattr(self, att, val)
            else:
                raise Exception("Unknown Parameter Type: Only list and dict are supported!")
        self.__dict__.update(kwds)

    def __str__(self):
        return "(%s)" % (",".join(["%s:%s" % (key, str(val)) for key, val in self.__dict__.items()]))

    def __repr__(self):
        return self.__str__()

    def hasMember(self, member):
        return member in self.__dict__

    def hasVar(self, var):
        return self.hasMember(var)

    def dumpAsJSon(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def loadFromJSon(jsonString):
        import json

        d = json.loads(jsonString)
        b = Bunch()
        for k, v in d.items():
            b.__dict__[k] = v
        return b


# class for storing a detected chromatographic peak pair.
# Thus, one ChromPeakPair instance stores two chromatographic peaks of a native and a labelled metabolite ion.
# Member variables are only set, if provides as parameters
class ChromPeakPair:
    def __init__(
        self,
        id=-1,
        fGroupID=-1,
        eicID=-1,
        massSpectrumID=-1,
        assignedName=-1,
        tracer=-1,
        tracerName="",
        mz=-1,
        lmz=-1,
        xCount=-1,
        loading=-1,
        ionMode="",
        NPeakCenter=-1,
        NPeakCenterMin=-1,
        NPeakScale=-1,
        NSNR=-1,
        NPeakArea=-1,
        NPeakAbundance=-1,
        LPeakCenter=-1,
        LPeakCenterMin=-1,
        LPeakScale=-1,
        LSNR=-1,
        LPeakArea=-1,
        LPeakAbundance=-1,
        heteroIsotoplogues={},
        assignedMZs=[],
        comments=[],
        artificialEICLShift=0,
        **args,
    ):
        argsUsed = 0

        self.id = id
        self.fGroupID = fGroupID
        self.eicID = eicID
        self.massSpectrumID = massSpectrumID
        self.assignedName = assignedName
        self.tracer = tracer
        self.tracerName = tracerName

        self.mz = mz
        self.lmz = lmz
        self.xCount = xCount
        self.loading = loading
        self.ionMode = ionMode

        self.NPeakCenter = NPeakCenter
        self.NPeakCenterMin = NPeakCenterMin
        self.NPeakScale = NPeakScale
        self.NSNR = NSNR
        self.NPeakArea = NPeakArea
        self.NPeakAbundance = NPeakAbundance

        self.LPeakCenter = LPeakCenter
        self.LPeakCenterMin = LPeakCenterMin
        self.LPeakScale = LPeakScale
        self.LSNR = LSNR
        self.LPeakArea = LPeakArea
        self.LPeakAbundance = LPeakAbundance

        self.heteroIsotopologues = heteroIsotoplogues or {}
        self.assignedMZs = assignedMZs or []

        self.comments = comments

        self.artificialEICLShift = artificialEICLShift

        if "tmz" in args:
            self.tmz = args["tmz"]
            argsUsed += 1

        if "peaksCorr" in args:
            self.peaksCorr = args["peaksCorr"]
            argsUsed += 1
        if "silRatios" in args:
            self.silRatios = args["silRatios"]
            argsUsed += 1
        if "peaksRatio" in args:
            self.peaksRatio = args["peaksRatio"]
            argsUsed += 1

        if "NXIC" in args:
            self.NXIC = args["NXIC"]
            argsUsed += 1
        if "LXIC" in args:
            self.LXIC = args["LXIC"]
            argsUsed += 1
        if "NXICSmoothed" in args:
            self.NXICSmoothed = args["NXICSmoothed"]
            argsUsed += 1
        if "LXICSmoothed" in args:
            self.LXICSmoothed = args["LXICSmoothed"]
            argsUsed += 1
        if "times" in args:
            self.times = args["times"]
            argsUsed += 1

        if "fDesc" in args:
            self.fDesc = args["fDesc"]
            argsUsed += 1
        if "adducts" in args:
            self.adducts = args["adducts"]
            argsUsed += 1
        if "heteroAtomsFeaturePairs" in args:
            self.heteroAtomsFeaturePairs = args["heteroAtomsFeaturePairs"]
            argsUsed += 1
        if "Ms" in args:
            self.Ms = args["Ms"]
            argsUsed += 1
        if "heteroAtoms" in args:
            self.heteroAtoms = args["heteroAtoms"]
            argsUsed += 1

        if "NBorderLeft" in args:
            self.NBorderLeft = args["NBorderLeft"]
            argsUsed += 1
        if "NBorderRight" in args:
            self.NBorderRight = args["NBorderRight"]
            argsUsed += 1
        if "LBorderLeft" in args:
            self.LBorderLeft = args["LBorderLeft"]
            argsUsed += 1
        if "LBorderRight" in args:
            self.LBorderRight = args["LBorderRight"]
            argsUsed += 1

        if "N_startRT" in args:
            self.N_startRT = args["N_startRT"]
            argsUsed += 1
        if "N_endRT" in args:
            self.N_endRT = args["N_endRT"]
            argsUsed += 1
        if "L_startRT" in args:
            self.L_startRT = args["L_startRT"]
            argsUsed += 1
        if "L_endRT" in args:
            self.L_endRT = args["L_endRT"]
            argsUsed += 1

        if "N_FWHM" in args:
            self.N_FWHM = args["N_FWHM"]
            argsUsed += 1
        if "L_FWHM" in args:
            self.L_FWHM = args["L_FWHM"]
            argsUsed += 1
        if "N_Baseline" in args:
            self.N_Baseline = args["N_Baseline"]
            argsUsed += 1
        if "L_Baseline" in args:
            self.L_Baseline = args["L_Baseline"]
            argsUsed += 1

        if "isotopeRatios" in args:
            self.isotopeRatios = args["isotopeRatios"]
            argsUsed += 1
        if "isotopologRatios" in args:
            self.isotopologRatios = args["isotopologRatios"]
            argsUsed += 1
        if "isoAreas" in args:
            self.isoAreas = args["isoAreas"]
            argsUsed += 1
        if "isoEnrichment" in args:
            self.isoEnrichment = args["isoEnrichment"]
            argsUsed += 1
        if "isoCount" in args:
            self.isoCount = args["isoCount"]
            argsUsed += 1
        if "mzDiffErrors" in args:
            self.mzDiffErrors = args["mzDiffErrors"]
            argsUsed += 1

        if "correlationsToOthers" in args:
            self.correlationsToOthers = args["correlationsToOthers"]
            argsUsed += 1

        if "new_FWHM_M" in args:
            self.new_FWHM_M = args["new_FWHM_M"]
            argsUsed += 1
        if "new_FWHM_Mp" in args:
            self.new_FWHM_Mp = args["new_FWHM_Mp"]
            argsUsed += 1
        if "new_Area_M" in args:
            self.new_Area_M = args["new_Area_M"]
            argsUsed += 1
        if "new_Area_Mp" in args:
            self.new_Area_Mp = args["new_Area_Mp"]
            argsUsed += 1
        if "new_SNR_M" in args:
            self.new_SNR_M = args["new_SNR_M"]
            argsUsed += 1
        if "new_SNR_Mp" in args:
            self.new_SNR_Mp = args["new_SNR_Mp"]
            argsUsed += 1

        assert argsUsed == len(args), "Not all agruments used %d %d" % (
            argsUsed,
            len(args),
        )

    def __str__(self):
        return "chrompeak: %f, %d (%f min), %d" % (
            self.mz,
            self.NPeakCenter,
            self.NPeakCenterMin / 60.0,
            self.id,
        )


# Faktorial of x
def fak(x):
    if x < 2:
        return 1.0
    return x * fak(x - 1)


# binomial coefficient n over k
def binom(n, k):
    return fak(n) / (fak(k) * fak(n - k))


# relative ratio of intensity for peak with a total of x 12-carbon atoms having s 13-carbon substitutions
# with p being the purity for 12-carbon and 1-p being the purity for 13-carbon
def getRatio(p, x, s):
    return (p ** (x - s)) * ((1.0 - p) ** s) * binom(x, s)


# normalized ratio to s=0:=1
def getNormRatio(p, x, s):
    return getRatio(p, x, s) / (p**x)


# Get most intense isotopologue for purity p and Cx
def getAtomAdd(p, x):
    m = 5
    minIndex, minValue = max(enumerate([getNormRatio(p, x, j) for j in range(m)]), key=itemgetter(1))
    return minIndex


# maps and eic to a certain reference time array.
# Used to reduce the number of scans in EICs before chromatographic alignment
def mapArrayToRefTimes(data, times, refTimes):
    data_arr = np.asarray(data, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    refTimes_arr = np.asarray(refTimes, dtype=np.float64)

    assert len(data_arr) == len(times_arr), "Data arrays have different length"

    # Use numpy's interp for efficient interpolation
    # This interpolates data values at refTimes based on times
    ref = np.interp(refTimes_arr, times_arr, data_arr)

    return ref


def mean(x, skipExtremes=0):
    x_arr = np.asarray(x)
    if len(x_arr) == 0:
        return None

    if skipExtremes > 0:
        x_arr = np.sort(x_arr)
        len(x_arr)
        x_arr = x_arr[int(len(x_arr) * skipExtremes) : int(len(x_arr) - len(x_arr) * skipExtremes)]

    return np.mean(x_arr)


def weightedMean(x, w):
    x_arr = np.asarray(x)
    w_arr = np.asarray(w)
    assert len(x_arr) == len(w_arr)

    if len(x_arr) == 0:
        return None

    w_normalized = w_arr / np.sum(w_arr)
    return np.sum(x_arr * w_normalized)


def sd(x):
    x_arr = np.asarray(x)
    if len(x_arr) <= 1:
        return None

    return np.std(x_arr)


def weightedSd(x, w):
    x_arr = np.asarray(x)
    w_arr = np.asarray(w)
    assert len(x_arr) == len(w_arr)

    if len(x_arr) <= 1:
        return None

    w_normalized = w_arr / np.max(w_arr)
    m = weightedMean(x_arr, w_arr)
    return np.sqrt(np.sum(w_normalized * np.power(x_arr - m, 2)) / len(x_arr))


def corr(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    assert len(a_arr) == len(b_arr)

    n = len(a_arr)

    if n < 2:
        return -1

    # Use numpy's corrcoef for efficient correlation calculation
    try:
        corr_matrix = np.corrcoef(a_arr, b_arr)
        if np.isnan(corr_matrix[0, 1]):
            return -1
        return corr_matrix[0, 1]
    except (RuntimeWarning, FloatingPointError):
        return -1


def getLastTimeBefore(times, refTime):
    return min(
        [(i, abs(s - refTime)) for i, s in zip(range(len(times)), times)],
        key=lambda x: x[1],
    )[0]


# returns non-connected sub-graphs in a graph
def getSubGraphs(nodes):
    used = []
    groups = []

    for k in nodes.keys():
        if k not in used:
            current = []
            group = []
            current.append(k)
            while len(current) > 0:
                cur = current.pop()
                if cur not in used:
                    group.append(cur)
                    used.append(cur)
                    current.extend(nodes[cur])
            groups.append(group)
    return groups


def getSubGraphsFromDictDict(nodes):
    used = []
    groups = []

    for k in nodes.keys():
        if k not in used:
            current = []
            group = []
            current.append(k)
            while len(current) > 0:
                cur = current.pop()
                if cur not in used:
                    group.append(cur)
                    used.append(cur)
                    current.extend(nodes[cur].keys())
            groups.append(group)
    return groups


# taken from http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
# use for a=["1", "2", "10", "11", "3"]
# natSort(a)
# for a=[("1", 1), ("2", 2), ("10", 3), ("11", 4), ("3", 5)]
# natSort(a, key=itemgetter(0))


def natSort(l, key=lambda ent: ent):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(ent, key=key):
        return [convert(c) for c in re.split("([0-9]+)", str(key(ent)))]

    # Convert to list first to handle Python 3 dict_keys, dict_values, etc.
    if not isinstance(l, list):
        l = list(l)
    l.sort(key=alphanum_key)
    return l


# HELPER METHOD used to calculate all possible (tupleSize-)combinations of elements
def _getCombinations(elements, tupleSize=2, start=0):
    if tupleSize > 1:
        j = []

        s = start
        while s < (len(elements) - tupleSize + 1):
            h = _getCombinations(elements, tupleSize - 1, start=s + 1)
            for i in h:
                a = [elements[s]]
                a.extend(i)
                j.append(a)
            s += 1

        return j

    elif tupleSize == 1:
        return [[x] for x in elements[start : (len(elements))]]


# method for calculating all possible (startTupleSize-)combinations of elements
def getXCombinations(elements, startTupleSize=2, endTupleSize=-1):
    if endTupleSize == -1:
        endTupleSize = len(elements)

    j = []
    for r in range(startTupleSize, endTupleSize + 1):
        j.extend(_getCombinations(elements, r))
    return j


# Processing-level constants: control which pipeline step(s) a group's files take part in.
GROUP_LEVEL_NORMAL = "normal"
GROUP_LEVEL_EXCLUDE_STEP1 = "exclude_step1"
GROUP_LEVEL_REINTEGRATION_ONLY = "reintegration_only"
GROUP_LEVEL_MSMS_ONLY = "msms_only"

# Order in which the levels are shown in the UI drop-down (also the natural "increasing exclusion" order)
GROUP_PROCESSING_LEVEL_ORDER = [
    GROUP_LEVEL_NORMAL,
    GROUP_LEVEL_EXCLUDE_STEP1,
    GROUP_LEVEL_REINTEGRATION_ONLY,
    GROUP_LEVEL_MSMS_ONLY,
]

# Human readable labels used in the group table's "Processing" combo box
GROUP_PROCESSING_LEVEL_LABELS = {
    GROUP_LEVEL_NORMAL: "Process normally",
    GROUP_LEVEL_EXCLUDE_STEP1: "Exclude step 1",
    GROUP_LEVEL_REINTEGRATION_ONLY: "Use only for re-integration",
    GROUP_LEVEL_MSMS_ONLY: "Do not process (use MSMS spectra only)",
}


# HELPER CLASS for storing a defined group
class SampleGroup:
    def __init__(
        self,
        name,
        files,
        minFound,
        omitFeatures,
        useForMetaboliteGrouping,
        removeAsFalsePositive,
        color,
        useAsMSMSTarget,
        processingLevel=GROUP_LEVEL_NORMAL,
    ):
        self.name = name
        self.files = files
        self.minFound = minFound
        self.omitFeatures = omitFeatures
        self.useForMetaboliteGrouping = useForMetaboliteGrouping
        self.removeAsFalsePositive = removeAsFalsePositive
        self.color = color
        self.useAsMSMSTarget = useAsMSMSTarget
        self.processingLevel = processingLevel if processingLevel in GROUP_PROCESSING_LEVEL_LABELS else GROUP_LEVEL_NORMAL


def is_int(s):
    try:
        int(s)
    except Exception:
        return False

    return True


def is_float(s):
    try:
        float(s)
    except Exception:
        return False

    return True


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


## from alex.forencich at https://stackoverflow.com/questions/2298339/standard-deviation-for-sqlite
class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return sqrt(self.S / (self.k - 2))


def getFileHash_sha1(filePath):
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open("anotherfile.txt", "rb") as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


########################################################################################################################
########################################################################################################################
########################################################################################################################
##############                                                                                            ##############
##############                      HELPER METHODS for FILE IO, SQLITE and PRINTING                       ##############
##############                                                                                            ##############
########################################################################################################################
########################################################################################################################
########################################################################################################################


def readTSVFileAsBunch(
    file,
    delim="\t",
    parseToNumbers=True,
    useColumns=None,
    omitFirstNRows=0,
    renameRows=None,
    commentChar="#",
):
    if renameRows is None:
        renameRows = {}

    # Convert string parameters to bytes for binary mode file operations
    delim_bytes = delim.encode("utf-8") if isinstance(delim, str) else delim
    commentChar_bytes = commentChar.encode("utf-8") if isinstance(commentChar, str) else commentChar

    with open(file, "rb") as fin:
        curRowi = 0
        headers = {}

        bunchs = []

        for line in fin:
            if omitFirstNRows > 0:
                continue
            if line.startswith(commentChar_bytes):
                continue

            hs = line.split(delim_bytes)
            hs = [h.strip().decode("utf-8", errors="replace") for h in hs]
            if curRowi == 0:
                for j, h in enumerate(hs):
                    if h in renameRows.keys():
                        headers[j] = renameRows[h]
                    else:
                        headers[j] = h
            else:
                b = Bunch(_addFollowing=list(headers.values()), _addFollowingDefaultValue="")
                for j, h in enumerate(hs):
                    if j not in headers:
                        continue
                    if useColumns is None or headers[j] in useColumns:
                        setattr(b, headers[j], h)

                bunchs.append(b)
            curRowi += 1

    headTypes = {}

    if parseToNumbers:
        for header in headers.values():
            headTypes[header] = "str"

            minValTypeFound = "int"
            for b in bunchs:
                attr = getattr(b, header)
                if minValTypeFound == "int" and is_int(attr):
                    minValTypeFound = "int"

                elif (minValTypeFound == "float" or (minValTypeFound == "int" and not is_int(attr))) and is_float(attr):
                    minValTypeFound = "float"

                else:
                    minValTypeFound = "str"

            if minValTypeFound != "str":
                for b in bunchs:
                    val = eval("%s('%s')" % (minValTypeFound, getattr(b, header)))
                    setattr(b, header, val)
                    headTypes[header] = minValTypeFound
    else:
        for header in headers.values():
            headTypes[header] = "str"

    return headTypes, bunchs


def readXLSXFileAsBunch(file, sheetName="", useColumns=None, omitFirstNRows=0):
    if sheetName == "":
        bn = "Sheet1"
        rf = bn.rfind(".")
        if rf > -1:
            sheetName = bn[:rf]
        else:
            sheetName = bn
    assert omitFirstNRows >= 0

    rb = open_workbook(file)

    ind = 0
    sheetInd = -1
    for s in rb.sheet_names():
        if s == sheetName:
            sheetInd = ind
        ind = ind + 1
    assert sheetInd != -1, "Sheet not found"
    sh = rb.sheet_by_name(sheetName)

    i = 0
    remainingCols = True
    headers = []
    colIDs = []
    while remainingCols:
        col = i
        row = omitFirstNRows
        try:
            if sh.cell(row, col).value is None or str(sh.cell(row, col).value) == "":
                remainingCols = False
                continue
        except Exception:
            remainingCols = False
            continue

        header = str(sh.cell(row, col).value)
        if useColumns is None or header in useColumns:
            headers.append(header)
            colIDs.append(i)
        i = i + 1

    rows = []
    r = 1 + omitFirstNRows
    allEmpty = False
    while not allEmpty:
        rowValues = []
        allEmpty = True
        for j in colIDs:
            col = j
            row = r
            try:
                if sh.cell(row, col).value is None or str(sh.cell(row, col).value) == "":
                    rowValues.append("")
                    j = j + 1
                    continue
            except Exception:
                rowValues.append("")
                j = j + 1
                continue
            rowValues.append(str(sh.cell(row, col).value))
            allEmpty = False
        if not allEmpty:
            rows.append(rowValues)

        r = r + 1

    dat = []
    for row in rows:
        b = Bunch()
        for j, head in enumerate(headers):
            setattr(b, head, row[j])
        dat.append(b)

    return headers, dat


def writeBunchsAsTSVFile(
    bunchs,
    outFile,
    delim="\t",
    writeAttributes=None,
    useAllAttributes=False,
    renameRows={},
):
    if writeAttributes is None:
        writeAttributes = []

        attributes = {}
        for b in bunchs:
            for att in b.__dict__.keys():
                if att not in attributes.keys():
                    attributes[att] = 0
                attributes[att] += 1

        for att in sorted(attributes.keys()):
            if attributes[att] == len(bunchs) or useAllAttributes:
                writeAttributes.append(att)

    with open(outFile, "w") as fo:
        fo.write(delim.join([att if att not in renameRows.keys() else renameRows[att] for att in writeAttributes]))
        fo.write("\n")
        for b in bunchs:
            fo.write(delim.join([str(getattr(b, att)) if hasattr(b, att) else "" for att in writeAttributes]))
            fo.write("\n")


def SQLSelectAsObject(curs_or_db, selectStatement, returnColumns=False, newObject=Bunch):
    """Select objects from a table. Works with both SQLite cursor and ParquetDB."""
    # Check if it's a ParquetDB object
    if hasattr(curs_or_db, "get_table"):
        # ParquetDB - parse the SELECT statement
        import re

        # Extract table name and columns from SELECT statement
        # Format: SELECT col1, col2 AS alias, ... FROM tablename WHERE ...
        match = re.search(r"SELECT\s+(.+?)\s+FROM\s+(\w+)", selectStatement, re.IGNORECASE | re.DOTALL)
        if not match:
            return

        cols_str = match.group(1).strip()
        table_name = match.group(2).strip()

        # Get the table
        df = curs_or_db.get_table(table_name)
        if df is None or len(df) == 0:
            return

        # Parse WHERE clause if present
        where_match = re.search(r"WHERE\s+(.+?)(?:ORDER BY|$)", selectStatement, re.IGNORECASE | re.DOTALL)
        if where_match:
            # This is a simplified WHERE parser - may need enhancement for complex queries
            where_match.group(1).strip()
            # For now, skip filtering as it requires complex parsing
            # Users should use execute_select method for complex queries
            pass

        # Parse column aliases
        columns_map = {}
        if cols_str.strip() != "*":
            col_parts = re.split(r",\s*", cols_str)
            for col_part in col_parts:
                as_match = re.search(r"(\S+)\s+AS\s+(\w+)", col_part, re.IGNORECASE)
                if as_match:
                    columns_map[as_match.group(2)] = as_match.group(1)
                else:
                    col_name = col_part.strip()
                    columns_map[col_name] = col_name
        else:
            columns_map = {col: col for col in df.columns}

        # Iterate through rows and create objects
        names = list(columns_map.keys())
        for row_dict in df.to_dicts():
            b = newObject()
            for alias, col_name in columns_map.items():
                if col_name in row_dict:
                    setattr(b, alias, row_dict[col_name])
                elif "." in col_name:  # Handle table.column syntax
                    actual_col = col_name.split(".")[-1]
                    if actual_col in row_dict:
                        setattr(b, alias, row_dict[actual_col])
            if returnColumns:
                yield names, b
            else:
                yield b
    else:
        # SQLite cursor (legacy support)
        curs_or_db.execute(selectStatement)
        names = list(map(lambda x: x[0], curs_or_db.description))
        for row in curs_or_db:
            b = newObject()
            for i in range(len(names)):
                setattr(b, names[i], row[i])
            if returnColumns:
                yield names, b
            else:
                yield b


def SQLgetSingleFieldFromOneRow(curs, selectStatement):
    curs.execute(selectStatement)
    for row in curs:
        return row[0]


def SQLInsert(curs_or_db, tableName, **kwargs):
    """Insert a row into a table. Works with both SQLite cursor and ParquetDB."""
    # Check if it's a ParquetDB object (has insert_row method) or SQLite cursor
    if hasattr(curs_or_db, "insert_row"):
        # ParquetDB
        curs_or_db.insert_row(tableName, kwargs)
    else:
        # SQLite cursor (legacy support)
        keys = list(kwargs.keys())
        vals = []

        for key in keys:
            val = kwargs[key]

            if isinstance(val, int) or isinstance(val, float):
                vals.append(val)
            else:
                vals.append("%s" % val)

        assert len(keys) == len(vals)

        parts = ["INSERT INTO"]
        parts.append(tableName)
        parts.append("(")

        needsComa = False
        for key in keys:
            if needsComa:
                parts.append(",")
            needsComa = True
            parts.append(key)

        parts.append(")")

        parts.append("VALUES (")

        needsComa = False
        for val in vals:
            if needsComa:
                parts.append(",")
            needsComa = True
            parts.append("?")

        parts.append(")")

        sqlCommand = " ".join(parts)
        curs_or_db.execute(sqlCommand, vals)


def writeObjectsAsSQLInsert(curs, toTable, objs, writeFields, fieldsMappingToColumns=None):
    if fieldsMappingToColumns is None:
        fieldsMappingToColumns = {}
    rowIDs = []
    for obj in objs:
        kwargs = {}
        for field in writeFields:
            param = field
            if field in fieldsMappingToColumns.keys():
                param = fieldsMappingToColumns[field]
            kwargs[param] = getattr(obj, field)
        SQLInsert(curs, toTable, **kwargs)
        rowIDs.append(curs.lastrowid)
    return rowIDs


def writeObjectAsSQLInsert(curs, toTable, obj, writeFields, fieldsMappingToColumns=None):
    if fieldsMappingToColumns is None:
        fieldsMappingToColumns
    return writeObjectsAsSQLInsert(curs, toTable, [obj], writeFields, fieldsMappingToColumns)[0]


def createTableFromBunch(
    tableName,
    bunchObject,
    cursor,
    useAttrs=None,
    primaryKeys=None,
    autoIncrements=None,
    ifNotExists=False,
):
    if useAttrs is None:
        useAttrs = list(bunchObject.__dict__.keys())
    if primaryKeys is None:
        primaryKeys = []
    if autoIncrements is None:
        autoIncrements = []

    sqlCommand = ["CREATE TABLE "]
    if ifNotExists:
        sqlCommand.append("IF NOT EXISTS ")
    sqlCommand.extend([tableName, " ("])

    needsComma = False
    for attr in useAttrs:
        typ = getattr(bunchObject, attr)
        if (type(typ) is type and typ is int) or (type(typ) is int):
            if needsComma:
                sqlCommand.append(", ")
            sqlCommand.extend([attr, " ", "INTEGER"])
            if attr in primaryKeys:
                sqlCommand.extend([" ", "PRIMARY KEY"])
            if attr in autoIncrements:
                sqlCommand.extend([" ", "AUTOINCREMENT"])
            needsComma = True
        elif (type(typ) is type and typ is float) or (type(typ) is float):
            if needsComma:
                sqlCommand.append(", ")
            sqlCommand.extend([attr, " ", "FLOAT"])
            if attr in primaryKeys:
                sqlCommand.extend([" ", "PRIMARY KEY"])
            if attr in autoIncrements:
                sqlCommand.extend([" ", "AUTOINCREMENT"])
            needsComma = True
        elif (type(typ) is type and typ is str) or (type(typ) is str) or True:
            if needsComma:
                sqlCommand.append(", ")
            sqlCommand.extend([attr, " ", "TEXT"])
            if attr in primaryKeys:
                sqlCommand.extend([" ", "PRIMARY KEY"])
            if attr in autoIncrements:
                sqlCommand.extend([" ", "AUTOINCREMENT"])
            needsComma = True

    sqlCommand.append(")")

    cursor.execute("".join(sqlCommand))


def printAsTable(heads, data, printInExcelFormat=False, excelSep="\t", sepEach=15):
    widths = [len(head) for head in heads]

    for row in data:
        if len(row) != len(heads):
            raise Exception("data not consistent")
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    if printInExcelFormat:
        row_format = excelSep.join("{%d}" % i for i in range(len(heads)))
    else:
        row_format = "  ".join("{%d:>%d}" % (i, widths[i]) for i in range(len(heads)))
    print(row_format.format(*heads))
    if not printInExcelFormat:
        hrow = row_format.replace(" ", "-").format(*["-" * widths[i] for i in range(len(widths))])
        print(hrow)

    j = 0
    for row in data:
        print(row_format.format(*row))
        j += 1
        if j == sepEach and not printInExcelFormat:
            print(hrow)
            j = 0


def printObjectsAsTable(objs, attrs, printInExcelFormat=False, excelSep="\t"):
    widths = [len(attr) for attr in attrs]

    for obj in objs:
        for j, attr in enumerate(attrs):
            if not hasattr(obj, attr):
                raise Exception("data not consistent")
            widths[j] = max(widths[j], len(str(getattr(obj, attr))))

    if printInExcelFormat:
        row_format = excelSep.join("{%d}" % i for i in range(len(attrs)))
    else:
        row_format = "  ".join("{%d:>%d}" % (i, widths[i]) for i in range(len(attrs)))
    print(row_format.format(*attrs))
    if not printInExcelFormat:
        hrow = row_format.replace(" ", "-").format(*["-" * widths[i] for i in range(len(widths))])
        print(hrow)

    j = 0
    for obj in objs:
        row = []
        for attr in attrs:
            row.append(getattr(obj, attr))
        print(row_format.format(*row))
        j += 1
        if j == 15 and not printInExcelFormat:
            print(hrow)
            j = 0


def printArraysAsTable(objs, positions, printInExcelFormat=False, excelSep="\t"):
    widths = [len(str(pos)) for pos in positions]

    for obj in objs:
        for j, pos in enumerate(positions):
            if len(obj) <= pos:
                raise Exception("data not consistent")
            widths[j] = max(widths[j], len(str(obj[pos])))

    if printInExcelFormat:
        row_format = excelSep.join("{%d}" % i for i in range(len(positions)))
    else:
        row_format = "  ".join("{%d:>%d}" % (i, widths[i]) for i in range(len(positions)))
    print(row_format.format(*positions))
    if not printInExcelFormat:
        hrow = row_format.replace(" ", "-").format(*["-" * widths[i] for i in range(len(widths))])
        print(hrow)

    j = 0
    for obj in objs:
        row = []
        for pos in positions:
            row.append(obj[pos])
        print(row_format.format(*row))
        j += 1
        if j == 15 and not printInExcelFormat:
            print(hrow)
            j = 0


def printDictDictAsTable(rowDict, colHeads=[], rowsOrd=None, printInExcelFormat=False, excelSep="\t"):
    if rowsOrd is None:
        rowsOrd = list(rowDict.keys())

    colHeads.insert(0, "")
    for row in rowDict.keys():
        if len(colHeads[0]) < len(row):
            colHeads[0] = row

    widths = [len(str(colHead)) for colHead in colHeads]

    for row in rowsOrd:
        cells = rowDict[row]
        for i in range(1, len(colHeads)):
            widths[i] = max(widths[i], len(str(cells[colHeads[i]])))

    colHeads[0] = ""

    if printInExcelFormat:
        row_format = excelSep.join("{%d}" % (i) for i in range(len(colHeads)))
    else:
        row_format = "  ".join("{%d:>%d}" % (i, widths[i]) for i in range(len(colHeads)))
    print(row_format.format(*colHeads))
    if not printInExcelFormat:
        hrow = row_format.replace(" ", "-").format(*["".join(["-" for j in range(widths[i])]) for i in range(len(widths))])
        print(hrow)

    j = 0
    for rowName in rowsOrd:
        row = []
        row.append(rowName)
        for i in range(1, len(colHeads)):
            row.append(rowDict[rowName][colHeads[i]])
        print(row_format.format(*row))
        j += 1
        if j == 4 and not printInExcelFormat:
            print(hrow)
            j = 0


if __name__ == "__main__" and True:
    import sqlite3

    conn = sqlite3.connect(":memory:")
    curs = conn.cursor()

    createTableFromBunch(
        tableName="foo",
        bunchObject=Bunch(id=int, name=str, address=str, sex=int),
        cursor=curs,
        primaryKeys=["id"],
        autoIncrements=["id"],
    )

    b = [
        Bunch(name="Peter", address="New York", sex=1),
        Bunch(name="Anna", address="Sydney", sex=2),
        Bunch(name="Norbert", address="Linz", sex=1),
    ]
    writeObjectsAsSQLInsert(curs, "foo", b, writeFields=["name", "address", "sex"])
    conn.commit()

    objs = [obj for obj in SQLSelectAsObject(curs, "select id, name, address, sex from foo")]
    printObjectsAsTable(objs, attrs=["id", "name", "address", "sex"])

    curs.close()
    conn.close()


def getDBSuffix():
    """Return the database suffix for MetExtract result files."""
    return ".meii"


def getDBFormat():
    """Return the database format for MetExtract result files."""
    return "parquet"


def add_sheet_to_excel(file, polars_df, sheet_name, overwrite=False):
    """
    Add a new sheet to an existing Excel file while preserving all other sheets.

    Args:
        file: Path to the Excel file
        polars_df: Polars DataFrame to write to the new sheet
        sheet_name: Name of the sheet to add/replace
    """

    db = PolarsDB.PolarsDB(file)
    if not db.has_table(sheet_name):
        # If the table does not exist, simply insert the DataFrame
        db.insert_table(sheet_name, polars_df)
    elif overwrite:
        db.set_table(sheet_name, polars_df)
    else:
        raise RuntimeError(f"Sheet '{sheet_name}' already exists in the Excel file. Use overwrite=True to replace it.")

    db.close()


@contextlib.contextmanager
def suppress_logs():
    # Store the current log handlers to restore them later
    original_log_handlers = logging.getLogger().handlers[:]

    # Remove all existing log handlers
    for handler in original_log_handlers:
        logging.getLogger().removeHandler(handler)

    yield

    # Restore the original log handlers after the tests
    for handler in original_log_handlers:
        logging.getLogger().addHandler(handler)

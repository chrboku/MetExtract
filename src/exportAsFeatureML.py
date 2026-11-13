import csv
from time import gmtime, strftime
from .MetExtractII_Main import MetExtractVersion
from .utils import Bunch


def _writeFeatureMLHeader(fileLineArray, features):
    fileLineArray.append('<?xml version="1.0" encoding="ISO-8859-1"?>')
    fileLineArray.append('<featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">')
    fileLineArray.append('	<dataProcessing completion_time="%s">' % (strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
    fileLineArray.append('		<software name="MetExtract II" version="%s" />' % (MetExtractVersion))
    fileLineArray.append('		<processingAction name="Feature pair detection" />')
    fileLineArray.append("	</dataProcessing>")
    fileLineArray.append('	<featureList count="%d">' % (len(features)))


def _writeFeatureMLFooter(fileLineArray):
    fileLineArray.append("	</featureList>")
    fileLineArray.append("</featureMap>")


def _writeFeatureMLToFile(fileLineArray, toFile):
    with open(toFile, "w") as fOut:
        for line in fileLineArray:
            fOut.write(line)
            fOut.write("\r\n")


def _parseFloats(value):
    """Parses a (possibly ';'-separated) string of floats, ignoring empty/invalid entries."""
    if value is None:
        return []
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if part == "":
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def _findSampleNames(columnNames):
    """Derives per-sample column name prefixes from the '<sample>_N_startRT' columns."""
    suffix = "_N_startRT"
    return [c[: -len(suffix)] for c in columnNames if c.endswith(suffix)]


def _averagePeakShape(get, sampleNames):
    """Averages the per-sample native/labeled start/apex/end RTs (only over samples with a
    detected peak) and returns them in seconds, or None where no data is available."""
    nStarts, nApexes, nEnds = [], [], []
    lStarts, lApexes, lEnds = [], [], []
    for s in sampleNames:
        nStarts.extend(_parseFloats(get(f"{s}_N_startRT")))
        nApexes.extend(_parseFloats(get(f"{s}_N_apexRT")))
        nEnds.extend(_parseFloats(get(f"{s}_N_endRT")))
        lStarts.extend(_parseFloats(get(f"{s}_L_startRT")))
        lApexes.extend(_parseFloats(get(f"{s}_L_apexRT")))
        lEnds.extend(_parseFloats(get(f"{s}_L_endRT")))

    def avgSec(vals):
        return (sum(vals) / len(vals)) * 60.0 if vals else None

    return (
        avgSec(nStarts),
        avgSec(nApexes),
        avgSec(nEnds),
        avgSec(lStarts),
        avgSec(lApexes),
        avgSec(lEnds),
    )


def _hShapePolygonPoints(bStart, bEnd, bLow, bHigh, bApex, tStart, tEnd, tLow, tHigh, tApex, rtBar):
    """Builds a single, non-self-intersecting polygon outline of a rotated 'H': a bottom
    leg rectangle (bStart..bEnd, bLow..bHigh) and a top leg rectangle (tStart..tEnd,
    tLow..tHigh), joined by a waist connecting their apex positions. Points are ordered
    so a simple path renderer (no hull re-computation) draws the shape correctly."""
    bwLeft = min(max(bStart, bApex - rtBar), bEnd)
    bwRight = max(min(bEnd, bApex + rtBar), bStart)
    twLeft = min(max(tStart, tApex - rtBar), tEnd)
    twRight = max(min(tEnd, tApex + rtBar), tStart)

    return [
        (bStart, bLow),
        (bEnd, bLow),
        (bEnd, bHigh),
        (bwRight, bHigh),
        (twRight, tLow),
        (tEnd, tLow),
        (tEnd, tHigh),
        (tStart, tHigh),
        (tStart, tLow),
        (twLeft, tLow),
        (bwLeft, bHigh),
        (bStart, bHigh),
    ]


def _featurePeakBounds(feature, rtPM):
    """Returns (nStart, nApex, nEnd, lStart, lApex, lEnd) in seconds for a feature.

    If the feature carries averaged per-sample peak boundaries (nStartRT/nEndRT/... and
    lStartRT/lEndRT/...), those are used. Otherwise falls back to a fixed +/- rtPM window
    around the feature's RT so features without per-sample data still render.
    """
    rt = feature.rt
    nStart = getattr(feature, "nStartRT", None)
    nEnd = getattr(feature, "nEndRT", None)
    nApex = getattr(feature, "nApexRT", None)
    lStart = getattr(feature, "lStartRT", None)
    lEnd = getattr(feature, "lEndRT", None)
    lApex = getattr(feature, "lApexRT", None)

    if nStart is None or nEnd is None or nEnd <= nStart:
        nStart, nEnd = rt - rtPM, rt + rtPM
    if nApex is None:
        nApex = rt
    if lStart is None or lEnd is None or lEnd <= lStart:
        lStart, lEnd = rt - rtPM, rt + rtPM
    if lApex is None:
        lApex = rt

    return nStart, nApex, nEnd, lStart, lApex, lEnd


def writeFeatureListToFeatureML(features, toFile, ppmPM=5, rtPM=0.25 * 60):
    """Writes two featureML files for the given feature pairs:

    a) <toFile>: the native (lower m/z) and labeled (higher m/z) chromatographic peaks are
       each shown as a rectangle spanning their (averaged, per-sample) detected start/end RTs.
    b) <toFile>_adapted.featureML: the same two peak rectangles, joined by a waist attaching
       to the apex RT of the native and the labeled peak, rendered as a single rotated-H hull.
    """
    writeFeatureListToFeatureMLStandard(features, toFile, ppmPM=ppmPM, rtPM=rtPM)
    writeFeatureListToFeatureMLAdapted(features, toFile.replace(".featureML", "_adapted.featureML"), ppmPM=3.0, rtPM=rtPM)


def writeFeatureListToFeatureMLStandard(features, toFile, ppmPM=5, rtPM=0.25 * 60):
    fileLineArray = []
    _writeFeatureMLHeader(fileLineArray, features)

    for feature in features:
        num = feature.id
        grpNum = feature.ogroup
        mz = feature.mz
        lmz = feature.lmz
        rt = feature.rt
        z = feature.charge
        name = feature.name
        xn = feature.Xn
        ionMode = feature.ionMode

        nStart, _nApex, nEnd, lStart, _lApex, lEnd = _featurePeakBounds(feature, rtPM)

        fileLineArray.append('		<feature id="%s">' % (num))
        fileLineArray.append('			<UserParam type="string" name="label" value="%s (Num: %s, OGroup: %s, MZ: %.5f, RT (min): %.2f, Charge: %d, Xn: %s, ionMode: %s)"/>' % (name, num, grpNum, mz, rt, z, xn, ionMode))
        fileLineArray.append('			<position dim="0">%f</position>' % (rt))
        fileLineArray.append('			<position dim="1">%f</position>' % (mz))
        fileLineArray.append("			<charge>%d</charge>" % (z))
        fileLineArray.append("			<intensity>1</intensity>")
        fileLineArray.append('			<quality dim="0">-1</quality>')
        fileLineArray.append('			<quality dim="1">-1</quality>')
        fileLineArray.append("			<overallquality>-1</overallquality>")
        # native (lower m/z) chromatographic peak, averaged start/end RT across samples
        fileLineArray.append('			<convexhull nr="1">')
        fileLineArray.append('				<pt x="%f" y="%f" />' % (nStart, mz * (1 - ppmPM / 1000000.0)))
        fileLineArray.append('				<pt x="%f" y="%f" />' % (nEnd, mz * (1 - ppmPM / 1000000.0)))
        fileLineArray.append('				<pt x="%f" y="%f" />' % (nStart, mz * (1 + ppmPM / 1000000.0)))
        fileLineArray.append('				<pt x="%f" y="%f" />' % (nEnd, mz * (1 + ppmPM / 1000000.0)))
        fileLineArray.append("			</convexhull>")
        # labeled (higher m/z) chromatographic peak, averaged start/end RT across samples
        fileLineArray.append('			<convexhull nr="2_lab">')
        fileLineArray.append('				<pt x="%f" y="%f" />' % (lStart, lmz * (1 - ppmPM / 1000000.0)))
        fileLineArray.append('				<pt x="%f" y="%f" />' % (lEnd, lmz * (1 - ppmPM / 1000000.0)))
        fileLineArray.append('				<pt x="%f" y="%f" />' % (lStart, lmz * (1 + ppmPM / 1000000.0)))
        fileLineArray.append('				<pt x="%f" y="%f" />' % (lEnd, lmz * (1 + ppmPM / 1000000.0)))
        fileLineArray.append("			</convexhull>")
        fileLineArray.append("		</feature>")

    _writeFeatureMLFooter(fileLineArray)
    _writeFeatureMLToFile(fileLineArray, toFile)


def writeFeatureListToFeatureMLAdapted(features, toFile, ppmPM=3.0, rtPM=0.25 * 60):
    fileLineArray = []
    _writeFeatureMLHeader(fileLineArray, features)

    for feature in features:
        num = feature.id
        grpNum = feature.ogroup
        mz = feature.mz
        lmz = feature.lmz
        rt = feature.rt
        z = feature.charge
        name = feature.name
        xn = feature.Xn
        ionMode = feature.ionMode

        nStart, nApex, nEnd, lStart, lApex, lEnd = _featurePeakBounds(feature, rtPM)
        nLow, nHigh = mz * (1 - ppmPM / 1000000.0), mz * (1 + ppmPM / 1000000.0)
        lLow, lHigh = lmz * (1 - ppmPM / 1000000.0), lmz * (1 + ppmPM / 1000000.0)

        # width of the connecting waist, relative to the narrower of the two peaks
        rtBar = max(min(nEnd - nStart, lEnd - lStart) * 0.05, rtPM * 0.02)

        # bottom/top legs of the rotated H are ordered by m/z, not by native/labeled
        if mz <= lmz:
            hPoints = _hShapePolygonPoints(nStart, nEnd, nLow, nHigh, nApex, lStart, lEnd, lLow, lHigh, lApex, rtBar)
        else:
            hPoints = _hShapePolygonPoints(lStart, lEnd, lLow, lHigh, lApex, nStart, nEnd, nLow, nHigh, nApex, rtBar)

        fileLineArray.append('		<feature id="%s">' % (num))
        fileLineArray.append('			<UserParam type="string" name="label" value="%s (Num: %s, OGroup: %s, MZ: %.5f, RT (min): %.2f, Charge: %d, Xn: %s, ionMode: %s)"/>' % (name, num, grpNum, mz, rt, z, xn, ionMode))
        fileLineArray.append('			<position dim="0">%f</position>' % (rt))
        fileLineArray.append('			<position dim="1">%f</position>' % (mz))
        fileLineArray.append("			<charge>%d</charge>" % (z))
        fileLineArray.append("			<intensity>1</intensity>")
        fileLineArray.append('			<quality dim="0">-1</quality>')
        fileLineArray.append('			<quality dim="1">-1</quality>')
        fileLineArray.append("			<overallquality>-1</overallquality>")
        # single rotated-H hull: native and labeled peak rectangles joined by an apex-to-apex waist
        fileLineArray.append('			<convexhull nr="1">')
        for px, py in hPoints:
            fileLineArray.append('				<pt x="%f" y="%f" />' % (px, py))
        fileLineArray.append("			</convexhull>")
        fileLineArray.append("		</feature>")

    _writeFeatureMLFooter(fileLineArray)
    _writeFeatureMLToFile(fileLineArray, toFile)


def convertMSMSoptFileToFeatureML(msmsFile, featureMLFile=None):
    if featureMLFile is None:
        featureMLFile = msmsFile + ".featureML"

    with open(msmsFile, "r", encoding="utf-8") as fIn:
        csvReader = csv.reader(fIn, delimiter=";", quotechar='"')

        headers = {}
        features = []

        for linei, row in enumerate(csvReader):
            if len(row) == 0:  # Skip empty lines
                continue
            if linei == 0:
                for colInd, header in enumerate(row):
                    headers[header] = colInd
            elif row[0].startswith("#"):
                pass
            else:
                b = Bunch(
                    id=row[headers["Comment"]],
                    ogroup="NA",
                    mz=float(row[headers["Mass [m/z]"]]),
                    rt=(float(row[headers["Start [min]"]]) * 60 + float(row[headers["End [min]"]]) * 60) / 2,
                    Xn=0,
                    lmz=float(row[headers["Mass [m/z]"]]),
                    charge=1,
                    name=row[headers["Comment"]],
                    ionMode=row[headers["Polarity"]],
                )
                features.append(b)

    writeFeatureListToFeatureML(features, featureMLFile, ppmPM=5.0, rtPM=0.25 * 60)


def convertMEMatrixToFeatureML(meMatrixFile, featureMLFile=None):
    if featureMLFile is None:
        featureMLFile = meMatrixFile.replace(".xlsx", ".tsv").replace(".tsv", ".txt").replace(".txt", "") + ".featureML"

    features = []

    # Check if file is Excel or TSV/CSV
    if meMatrixFile.endswith(".xlsx"):
        # Use polars to read Excel file
        import polars as pl

        df = pl.read_excel(meMatrixFile, sheet_name="Sheet1")
        sampleNames = _findSampleNames(df.columns)

        # Convert dataframe to list of features
        for row in df.iter_rows(named=True):
            nStart, nApex, nEnd, lStart, lApex, lEnd = _averagePeakShape(row.get, sampleNames)
            b = Bunch(
                id=str(row["Num"]),
                ogroup=str(row["OGroup"]),
                mz=float(row["MZ"]),
                rt=float(row["RT"]) * 60,
                Xn=int(row["Xn"]),
                lmz=float(row["L_MZ"]),
                charge=int(row["Charge"]),
                name=str(row["Num"]),
                ionMode=str(row["Ionisation_Mode"]),
                nStartRT=nStart,
                nApexRT=nApex,
                nEndRT=nEnd,
                lStartRT=lStart,
                lApexRT=lApex,
                lEndRT=lEnd,
            )
            features.append(b)
    else:
        # Original CSV/TSV reading logic
        with open(meMatrixFile, "r", encoding="utf-8") as fIn:
            csvReader = csv.reader(fIn, delimiter="\t", quotechar='"')

            headers = {}
            sampleNames = []
            for linei, row in enumerate(csvReader):
                if len(row) == 0:  # Skip empty lines
                    continue
                if linei == 0:
                    for colInd, header in enumerate(row):
                        headers[header] = colInd
                    sampleNames = _findSampleNames(headers.keys())
                elif row[0].startswith("#"):
                    pass
                else:
                    rowGet = lambda col: row[headers[col]] if col in headers else None  # noqa: E731
                    nStart, nApex, nEnd, lStart, lApex, lEnd = _averagePeakShape(rowGet, sampleNames)
                    b = Bunch(
                        id=row[headers["Num"]],
                        ogroup=row[headers["OGroup"]],
                        mz=float(row[headers["MZ"]]),
                        rt=float(row[headers["RT"]]) * 60,
                        Xn=int(row[headers["Xn"]]),
                        lmz=float(row[headers["L_MZ"]]),
                        charge=int(row[headers["Charge"]]),
                        name=row[headers["Num"]],
                        ionMode=row[headers["Ionisation_Mode"]],
                        nStartRT=nStart,
                        nApexRT=nApex,
                        nEndRT=nEnd,
                        lStartRT=lStart,
                        lApexRT=lApex,
                        lEndRT=lEnd,
                    )
                    features.append(b)

    writeFeatureListToFeatureML(features, featureMLFile, ppmPM=5.0, rtPM=0.25 * 60)


def convertMEMatrixToFeatureMLSepPolarities(
    meMatrixFile,
    sheet_name,
    featureMLFile=None,
    postfix="",
    numCol="Num",
    ogrpCol="OGroup",
    mzCol="MZ",
    rtCol="RT",
    xnCol="Xn",
    lmzCol="L_MZ",
    chargeCol="Charge",
    nameCol="Num",
    ionModeCol="Ionisation_Mode",
    delimiter="\t",
):
    if featureMLFile is None:
        featureMLFile = meMatrixFile.replace(".xlsx", ".tsv").replace(".tsv", ".txt").replace(".txt", "") + ".featureML"

    features = {"+": [], "-": []}

    # Check if file is Excel or TSV/CSV
    if meMatrixFile.endswith(".xlsx"):
        # Use polars to read Excel file
        import polars as pl

        df = pl.read_excel(meMatrixFile, sheet_name=sheet_name)
        sampleNames = _findSampleNames(df.columns)

        # Convert dataframe to list of features
        for row in df.iter_rows(named=True):
            nStart, nApex, nEnd, lStart, lApex, lEnd = _averagePeakShape(row.get, sampleNames)
            b = Bunch(
                id=str(row[numCol]),
                ogroup=str(row[ogrpCol]),
                mz=float(row[mzCol]),
                rt=float(row[rtCol]) * 60,
                Xn=str(row[xnCol]),
                lmz=float(row[lmzCol]),
                charge=int(row[chargeCol]),
                name=str(row[nameCol]),
                ionMode=str(row[ionModeCol]),
                nStartRT=nStart,
                nApexRT=nApex,
                nEndRT=nEnd,
                lStartRT=lStart,
                lApexRT=lApex,
                lEndRT=lEnd,
            )
            features[b.ionMode].append(b)
    else:
        # Original CSV/TSV reading logic
        with open(meMatrixFile, "r") as fIn:
            csvReader = csv.reader(fIn, delimiter=delimiter, quotechar='"')

            headers = {}
            sampleNames = []
            for linei, row in enumerate(csvReader):
                if len(row) == 0:  # Skip empty lines
                    continue
                if linei == 0:
                    for colInd, header in enumerate(row):
                        headers[header] = colInd
                    sampleNames = _findSampleNames(headers.keys())
                elif row[0].startswith("#"):
                    pass
                else:
                    rowGet = lambda col: row[headers[col]] if col in headers else None  # noqa: E731
                    nStart, nApex, nEnd, lStart, lApex, lEnd = _averagePeakShape(rowGet, sampleNames)
                    b = Bunch(
                        id=row[headers[numCol]],
                        ogroup=row[headers[ogrpCol]],
                        mz=float(row[headers[mzCol]]),
                        rt=float(row[headers[rtCol]]) * 60,
                        Xn=row[headers[xnCol]],
                        lmz=float(row[headers[lmzCol]]),
                        charge=int(row[headers[chargeCol]]),
                        name=row[headers[nameCol]],
                        ionMode=row[headers[ionModeCol]],
                        nStartRT=nStart,
                        nApexRT=nApex,
                        nEndRT=nEnd,
                        lStartRT=lStart,
                        lApexRT=lApex,
                        lEndRT=lEnd,
                    )
                    features[b.ionMode].append(b)

    if len(features["-"]) > 0:
        writeFeatureListToFeatureML(
            features["-"],
            featureMLFile.replace(".featureML", "_negMode%s.featureML" % (postfix)),
            ppmPM=5.0,
            rtPM=0.25 * 60,
        )
    if len(features["+"]) > 0:
        writeFeatureListToFeatureML(
            features["+"],
            featureMLFile.replace(".featureML", "_posMode%s.featureML" % (postfix)),
            ppmPM=5.0,
            rtPM=0.25 * 60,
        )


if __name__ == "__main__":
    convertMEMatrixToFeatureMLSepPolarities(
        "H:/180713_382_Labelboxpaper/Comparison_MZMine_MetExtractII/mzmine2results.csv",
        None,
        numCol="row ID",
        ogrpCol="row ID",
        mzCol="row m/z",
        rtCol="row retention time",
        xnCol="row ID",
        lmzCol="row m/z",
        chargeCol="row ID",
        nameCol="row ID",
        ionModeCol="row ionmode",
        delimiter=";",
    )

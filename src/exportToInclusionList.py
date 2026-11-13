import csv
import os
import pandas as pd


def writeIQXInclusionList(rows, toFile):
    """Write an IQ-X style inclusion list (comma-separated CSV).

    *rows* is a list of dicts with keys: compound, mz, tstart, tstop, intensityThreshold.
    """
    with open(toFile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        writer.writerow(["Compound", "m/z", "t start (min)", "t stop (min)", "Intensity Threshold"])
        for r in rows:
            writer.writerow(
                [
                    r["compound"],
                    "%.5f" % r["mz"],
                    "%.3f" % r["tstart"],
                    "%.3f" % r["tstop"],
                    r["intensityThreshold"],
                ]
            )


def writeQExactiveInclusionList(rows, toFile):
    """Write a QExactive style inclusion list (comma-separated CSV).

    *rows* is a list of dicts with keys: mz, polarity, tstart, tstop.
    All other columns are left empty.
    """
    headers = [
        "Mass [m/z]",
        "Formula [M]",
        "Formula type",
        "Species",
        "CS [z]",
        "Polarity",
        "Start [min]",
        "End [min]",
        "(N)CE",
        "MSXID",
        "Comment",
    ]
    data = []
    for r in rows:
        data.append(
            {
                "Mass [m/z]": "%.5f" % r["mz"],
                "Formula [M]": "",
                "Formula type": "",
                "Species": "",
                "CS [z]": "",
                "Polarity": r["polarity"],
                "Start [min]": "%.3f" % r["tstart"],
                "End [min]": "%.3f" % r["tstop"],
                "(N)CE": "",
                "MSXID": "",
                "Comment": "",
            }
        )
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(toFile, index=False, sep=",", lineterminator="\n")


def writeInclusionList(mzs, rtStarts, rtEnds, pols, toFile, comments=None):
    with open(toFile, "w") as tsvfile:
        writer = csv.writer(tsvfile, delimiter=",", lineterminator="\n")
        writer.writerow(
            [
                "Mass [m/z]",
                "Formula [M]",
                "Formula type",
                "Species",
                "CS [z]",
                "Polarity",
                "Start [min]",
                "End [min]",
                "(N)CE",
                "MSXID",
                "Comment",
            ]
        )
        for ind in range(len(mzs)):
            writer.writerow(
                [
                    str(mzs[ind]),
                    "",
                    "",
                    "",
                    "",
                    pols[ind],
                    str(rtStarts[ind]),
                    rtEnds[ind],
                    "",
                    "",
                    "" if comments is None else comments[ind],
                ]
            )


def genInclusionForFolder(processPath, rtPlusMinusMin=0.2):
    for file in os.listdir(processPath):
        if file.endswith(".mzXML.tsv"):
            print(os.path.basename(file))
            file = os.path.abspath(os.path.join(processPath, file))

            for mode in ["pos", "neg"]:
                df = pd.read_csv(file, sep="\t")

                modeS = "+" if mode == "pos" else "-"
                modeL = "positive" if mode == "pos" else "negative"

                df = df[df["Ionisation_Mode"] == modeS]

                mzs = [i for i in df["MZ"]] + [i for i in df["L_MZ"]]
                rtStarts = [i - rtPlusMinusMin for i in df["RT"]] + [i - rtPlusMinusMin for i in df["RT"]]
                rtEnds = [i + rtPlusMinusMin for i in df["RT"]] + [i + rtPlusMinusMin for i in df["RT"]]
                pols = [modeL for i in df["MZ"]] + [modeL for i in df["MZ"]]
                comments = ["Native" for i in df["MZ"]] + ["U13C" for i in df["MZ"]]

                writeInclusionList(
                    mzs,
                    rtStarts,
                    rtEnds,
                    pols,
                    toFile=os.path.join(
                        os.path.dirname(file),
                        "InclusionList_NativeU13C_%s_%s.csv"
                        % (
                            modeL,
                            os.path.basename(file).replace(".mzXML", "").replace(".tsv", ""),
                        ),
                    ),
                    comments=comments,
                )

                mzs = [i for i in df["MZ"]]
                rtStarts = [i - rtPlusMinusMin for i in df["RT"]]
                rtEnds = [i + rtPlusMinusMin for i in df["RT"]]
                pols = [modeL for i in df["MZ"]]
                comments = ["Native" for i in df["MZ"]]

                writeInclusionList(
                    mzs,
                    rtStarts,
                    rtEnds,
                    pols,
                    toFile=os.path.join(
                        os.path.dirname(file),
                        "InclusionList_Native_%s_%s.csv"
                        % (
                            modeL,
                            os.path.basename(file).replace(".mzXML", "").replace(".tsv", ""),
                        ),
                    ),
                    comments=comments,
                )

                mzs = [i for i in df["L_MZ"]]
                rtStarts = [i - rtPlusMinusMin for i in df["RT"]]
                rtEnds = [i + rtPlusMinusMin for i in df["RT"]]
                pols = [modeL for i in df["MZ"]]
                comments = ["Native" for i in df["MZ"]]

                writeInclusionList(
                    mzs,
                    rtStarts,
                    rtEnds,
                    pols,
                    toFile=os.path.join(
                        os.path.dirname(file),
                        "InclusionList_U13C_%s_%s.csv"
                        % (
                            modeL,
                            os.path.basename(file).replace(".mzXML", "").replace(".tsv", ""),
                        ),
                    ),
                    comments=comments,
                )


processPath = r"N:\iBAM\Christina\Inclusions lists"
rtPlusMinusMin = 0.2

if __name__ == "__main__":
    genInclusionForFolder(processPath, rtPlusMinusMin)

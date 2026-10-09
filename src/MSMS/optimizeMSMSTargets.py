import random
import time
from copy import deepcopy
from ..TableUtils import TableUtils
from ..utils import Bunch


class OptimizeMSMSTargetList:
    def __init__(self):
        self.MSMSTargets = []

    def addTarget(self, num, mz, rt, ionMode, abundancesInFiles=None):
        self.MSMSTargets.append(
            Bunch(
                num=num,
                mz=mz,
                rt=rt,
                ionMode=ionMode,
                abundancesInFiles=abundancesInFiles,
                first="",
                firstCounts="",
                second="",
                secondCounts="",
                third="",
                thirdCounts="",
            )
        )

    def readTargetsFromFile(self, file, samplesToUse):
        table = TableUtils.readFile(file)
        cols = ["Num", "MZ", "L_MZ", "RT", "Ionisation_Mode"]
        for samp in samplesToUse:
            cols.append("%s_Abundance_N" % samp)
            cols.append("%s_Abundance_L" % samp)

        for row in table.getData(cols=cols):
            abundancesInFiles = {}
            for sampi, samp in enumerate(samplesToUse):
                abundancesInFiles[samp] = row[sampi * 2 + 5]

            self.addTarget(
                num=str(row[0]) + "_N",
                mz=row[1],
                rt=row[3],
                ionMode="positive" if row[4] == "+" else "negative",
                abundancesInFiles=abundancesInFiles,
            )

        for row in table.getData(cols=cols):
            abundancesInFiles = {}
            for sampi, samp in enumerate(samplesToUse):
                abundancesInFiles[samp] = row[sampi * 2 + 1 + 5]

            self.addTarget(
                num=str(row[0]) + "_L",
                mz=row[2],
                rt=row[3],
                ionMode="positive" if row[4] == "+" else "negative",
                abundancesInFiles=abundancesInFiles,
            )

    def getMostAbundantFileList(self, minCounts=1):
        for target in self.MSMSTargets:
            samps = [(sampName, target.abundancesInFiles[sampName]) for sampName in target.abundancesInFiles.keys()]
            samps = [s for s in samps if s[1] >= minCounts and s[1] != ""]
            samps = sorted(samps, key=lambda x: x[1], reverse=True)

            if len(samps) > 0:
                target.first, target.firstCounts = samps[0]
            else:
                target.first = ""
                target.firstCounts = ""
            if len(samps) > 1:
                target.second, target.secondCounts = samps[1]
            else:
                target.second = ""
                target.secondCounts = ""
            if len(samps) > 2:
                target.third, target.thirdCounts = samps[2]
            else:
                target.third = ""
                target.thridCounts = ""

    def writeTargetsToFile(self, fromFile, toFile):
        table = TableUtils.readFile(fromFile)
        table.addColumn(col="MSMS_1st_N", colType="TEXT", defaultValue="")
        table.addColumn(col="MSMS_1st_Abundance_N", colType="TEXT", defaultValue="")

        table.addColumn(col="MSMS_2nd_N", colType="TEXT", defaultValue="")
        table.addColumn(col="MSMS_2nd_Abundance_N", colType="TEXT", defaultValue="")

        table.addColumn(col="MSMS_3rd_N", colType="TEXT", defaultValue="")
        table.addColumn(col="MSMS_3rd_Abundance_N", colType="TEXT", defaultValue="")

        table.addColumn(col="MSMS_1st_L", colType="TEXT", defaultValue="")
        table.addColumn(col="MSMS_1st_Abundance_L", colType="TEXT", defaultValue="")

        table.addColumn(col="MSMS_2nd_L", colType="TEXT", defaultValue="")
        table.addColumn(col="MSMS_2nd_Abundance_L", colType="TEXT", defaultValue="")

        table.addColumn(col="MSMS_3rd_L", colType="TEXT", defaultValue="")
        table.addColumn(col="MSMS_3rd_Abundance_L", colType="TEXT", defaultValue="")

        for target in self.MSMSTargets:
            if "_N" in target.num:
                table.setData(
                    cols=[
                        "MSMS_1st_N",
                        "MSMS_1st_Abundance_N",
                        "MSMS_2nd_N",
                        "MSMS_2nd_Abundance_N",
                        "MSMS_3rd_N",
                        "MSMS_3rd_Abundance_N",
                    ],
                    vals=[
                        target.first,
                        target.firstCounts,
                        target.second,
                        target.secondCounts,
                        target.third,
                        target.thirdCounts,
                    ],
                    where="Num='%s'" % str(target.num).replace("_N", ""),
                )
            elif "_L" in target.num:
                table.setData(
                    cols=[
                        "MSMS_1st_L",
                        "MSMS_1st_Abundance_L",
                        "MSMS_2nd_L",
                        "MSMS_2nd_Abundance_L",
                        "MSMS_3rd_L",
                        "MSMS_3rd_Abundance_L",
                    ],
                    vals=[
                        target.first,
                        target.firstCounts,
                        target.second,
                        target.secondCounts,
                        target.third,
                        target.thirdCounts,
                    ],
                    where="Num='%s'" % str(target.num).replace("_L", ""),
                )

        TableUtils.saveFile(table, fType="tsv", file=toFile)
        # exportAsFeatureML.convertMSMSoptFileToFeatureML(toFile)

    def generateMSMSLists(
        self,
        samplesToUse,
        fileTo,
        minCounts=1000000,
        rtPlusMinus=0.25,
        maxParallelTargets=5,
        numberOfFiles=2,
        noffsprings=20,
        permCount=5,
        ngenerations=500,
        showDebugPlot=True,
        pwSetText=None,
        pwSetMax=None,
        pwSetValue=None,
    ):
        useTargets = {}

        for target in self.MSMSTargets:
            target.abundancesInFiles
            if any([target.abundancesInFiles[samp] != "" and target.abundancesInFiles[samp] >= minCounts for samp in target.abundancesInFiles.keys()]):
                useTargets[target.num] = target

        mat = [[0 for i in range(len(samplesToUse) * numberOfFiles)] for j in range(len(useTargets))]
        matRowNames = [useTargets[targetNum].num for targetNum in useTargets.keys()]
        matColNames = []
        for t in range(numberOfFiles):
            matColNames.extend([samp + "_" + str(t + 1) for samp in samplesToUse])

        j = 0
        for rowi in range(len(matRowNames)):
            mat[rowi][j] = 1
            j = (j + 1) % len(matColNames)

        print("There will be", len(useTargets), "ions used for the optimization")

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.show(block=False)

        startProc = time.time()

        if pwSetMax is not None:
            pwSetMax(ngenerations)

        gens = []
        scores = []
        score = None
        startScore = None
        for geni in range(ngenerations):
            curPermCount = int(permCount * 1.0 * (ngenerations - geni) / ngenerations)
            # curPermCount=permCount

            prevScore = score
            score = self.calculateScore(
                mat,
                matRowNames,
                matColNames,
                useTargets,
                minCounts=minCounts,
                rtPlusMinus=rtPlusMinus,
                maxParallelTargets=maxParallelTargets,
                numberOfFiles=numberOfFiles,
            )

            if prevScore is None:
                prevScore = score
            if startScore is None:
                startScore = score

            gens.append(geni)
            scores.append(score)

            ax.clear()
            ax.plot(gens, scores)

            print(
                "Generation:",
                "%s(/%d)" % (geni, ngenerations),
                "current score: %.2E" % score,
                "(improvement to start score of %.1f%%)" % (abs(score / startScore - 1.0) * 100.0),
                "(after %.1f minutes)" % ((time.time() - startProc) / 60.0),
            )
            if pwSetText is not None:
                pwSetText(
                    " ".join(
                        [
                            "Generation:",
                            "%s(/%d)" % (geni, ngenerations),
                            "current score: %.2E" % score,
                            "(after %.1f minutes)" % ((time.time() - startProc) / 60.0),
                        ]
                    )
                )
            if pwSetValue is not None:
                pwSetValue(geni)

            bestOffspring = None
            bestOffspringScore = -10000000000000000000000
            for offspringi in range(noffsprings):
                newMat = self.permMatrix(deepcopy(mat), permCount=curPermCount)
                offspringScore = self.calculateScore(
                    newMat,
                    matRowNames,
                    matColNames,
                    useTargets,
                    minCounts=minCounts,
                    rtPlusMinus=rtPlusMinus,
                    maxParallelTargets=maxParallelTargets,
                    numberOfFiles=numberOfFiles,
                )
                # print("    offspring score is", offspringScore)

                if offspringScore > bestOffspringScore:
                    bestOffspring = newMat
                    bestOffspringScore = offspringScore

            # print("  Best offspring is", bestOffspringScore)

            if bestOffspringScore > score and bestOffspring is not None:
                mat = bestOffspring

            fig.tight_layout()
            fig.canvas.draw()

        # print("Finished.. improvement best score (%.2E) to start score (%.2E) of %.1f%%)"%(scoreD, startScore, (score/startScore-1.)*100.))
        print("Finished")
        plt.show()

        ## export target lists
        for samp in matColNames:
            samplei = matColNames.index(samp)
            sampOri = samp[: samp.rfind("_")]

            writeFile = False
            for rowi in range(len(matRowNames)):
                rowNum = matRowNames[rowi]
                isInSample = mat[rowi][samplei] == 1

                if isInSample:
                    writeFile = True
                    break

            if writeFile:
                with open(fileTo.replace(".tsv", "__MSMSTargets_%s.tsv" % samp), "w") as fout:
                    print(
                        "Writing file..",
                        fileTo.replace(".tsv", "__MSMSTargets_%s.tsv" % samp),
                    )
                    fout.write(
                        "\t".join(
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
                        + "\n"
                    )

                    for rowi in range(len(matRowNames)):
                        rowNum = matRowNames[rowi]
                        isInSample = mat[rowi][samplei] == 1

                        if isInSample:
                            target = useTargets[rowNum]
                            fout.write(
                                "\t".join(
                                    [
                                        str(t)
                                        for t in [
                                            target.mz,
                                            "",
                                            "",
                                            "",
                                            "",
                                            target.ionMode,
                                            target.rt - rtPlusMinus,
                                            target.rt + rtPlusMinus,
                                            "",
                                            "",
                                            "Num: %s, Abundance: %s"
                                            % (
                                                rowNum,
                                                target.abundancesInFiles[sampOri],
                                            ),
                                        ]
                                    ]
                                )
                                + "\n"
                            )

    def permMatrix(self, mat, permCount=25):
        for permi in range(permCount):
            rowi = random.randint(0, len(mat) - 1)
            row = mat[rowi]

            for coli in range(len(row)):
                mat[rowi][coli] = 0

            coli = random.randint(0, len(row) - 1)
            mat[rowi][coli] = 1

        return mat

    def calculateScore(
        self,
        mat,
        matRowNames,
        matColNames,
        useTargets,
        minCounts,
        rtPlusMinus,
        maxParallelTargets,
        numberOfFiles,
    ):
        score = 0

        for matRowi in range(len(matRowNames)):
            targetNum = matRowNames[matRowi]
            samplei = mat[matRowi].index(1)
            sampleName = matColNames[samplei]
            sampleName = sampleName[: sampleName.rfind("_")]
            sampleAbundance = useTargets[targetNum].abundancesInFiles[sampleName]

            if sampleAbundance != "":
                score = score + sampleAbundance
            else:
                score = score - 20 * minCounts
        if True:
            for samplei in range(len(matColNames)):
                for matRowi in range(len(matRowNames)):
                    targetiRT = useTargets[matRowNames[matRowi]].rt
                    targetiOverlaps = 0
                    targetiOverlapSum = 0
                    if mat[matRowi][samplei] == 1:
                        for matRowj in range(len(matRowNames)):
                            if matRowj < matRowi and mat[matRowj][samplei] == 1:
                                targetjRT = useTargets[matRowNames[matRowj]].rt

                                rtOverlap = getRTOverlap(targetiRT, targetjRT, rtPlusMinus)

                                if rtOverlap > 0:
                                    targetiOverlaps = targetiOverlaps + 1
                                    targetiOverlapSum = targetiOverlapSum + rtOverlap

                    if targetiOverlaps > maxParallelTargets:
                        score = score - rtOverlap * 20 * minCounts

        return score


def getRTOverlap(rta, rtb, pmWindow):
    if rtb < rta:
        t = rta
        rta = rtb
        rtb = t

    if (rta + pmWindow) > (rtb - pmWindow):
        return (rta + pmWindow) - (rtb - pmWindow)
    else:
        return 0


if __name__ == "__main__":
    opt = OptimizeMSMSTargetList()

    samplesToUse = [
        "posneg_Blank_1",
        "posneg_Blank_2",
        "posneg_12CKarurDON_13CKarurDON_Rep1",
        "posneg_12CKarurDON_13CKarurMock_Rep1",
        "posneg_12CKarurDON_13CKarurMock_Rep2",
        "posneg_12CKarurDON_13CKarurMock_Rep3",
        "posneg_12CKarurMock_13CKarurMock_Rep1",
    ]

    opt.readTargetsFromFile(
        file="H:/180713_382_Labelboxpaper/EVAL_PrimaryDataEvaluation_FastPolaritySwitching/results_C_FP.tsv",
        samplesToUse=samplesToUse,
    )

    opt.getMostAbundantFileList(minCounts=1000000)

    opt.writeTargetsToFile(
        "H:/180713_382_Labelboxpaper/EVAL_PrimaryDataEvaluation_FastPolaritySwitching/results_C_FP.tsv",
        "H:/180713_382_Labelboxpaper/EVAL_PrimaryDataEvaluation_FastPolaritySwitching/results_C_FP_MSMS.tsv",
    )

    opt.generateMSMSLists(
        samplesToUse,
        fileTo="H:/180713_382_Labelboxpaper/EVAL_PrimaryDataEvaluation_FastPolaritySwitching/results_C_FP_MSMS.tsv",
        minCounts=1000000,
    )

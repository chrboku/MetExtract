# a class used to parse chemical formulas
# e.g. the formula C6H12O6 will be parsed to a dictionary {'H':12, 'C':6, 'O':6}
# NOTE: different isotopes may be specified as [13C]C5H12O6
# Charged formulas are supported: C15H20O6+, C15H20O6+2, C15H20O6++, C15H20O6-, etc.
from __future__ import absolute_import, division, print_function
import re

# Mass of an electron in Daltons
ELECTRON_MASS = 0.000548579909


def _parse_charge_suffix(formula):
    """Parse charge notation from the end of a formula string.

    Returns (clean_formula, charge) where charge is an integer.
    Handles: +, ++, +N, -, --, -N (N is a positive integer).
    Examples:
        'C15H20O6+'  -> ('C15H20O6', 1)
        'C15H20O6+2' -> ('C15H20O6', 2)
        'C15H20O6++' -> ('C15H20O6', 2)
        'C15H20O6-'  -> ('C15H20O6', -1)
        'C15H20O6-2' -> ('C15H20O6', -2)
        'C6H12O6'    -> ('C6H12O6', 0)
    """
    formula = formula.strip()
    m = re.search(r"(\+{2,}|\-{2,}|\+\d+|\-\d+|\+|\-)$", formula)
    if m:
        charge_str = m.group(1)
        if charge_str[0] == "+":
            if len(charge_str) == 1:
                charge = 1
            elif charge_str[1:].isdigit():
                charge = int(charge_str[1:])
            else:
                charge = len(charge_str)
        else:
            if len(charge_str) == 1:
                charge = -1
            elif charge_str[1:].isdigit():
                charge = -int(charge_str[1:])
            else:
                charge = -len(charge_str)
        return formula[: m.start()], charge
    return formula, 0


class formulaTools:
    def __init__(self, elemDetails=None):
        if elemDetails is None:
            self.elemDetails = {}
            #                Element    Name            short Neutrons Mass    Abundance
            self.elemDetails["Al"] = ["Aluminum", "Al", 27, 26.981541, 1.00]
            self.elemDetails["Sb"] = ["Antimony", "Sb", 121, 120.903824, 0.573]
            self.elemDetails["Ar"] = ["Argon", "Ar", 40, 39.962383, 0.996]
            self.elemDetails["As"] = ["Arsenic", "As", 75, 74.921596, 1.00]
            self.elemDetails["Ba"] = ["Barium", "Ba", 138, 137.905236, 0.717]
            self.elemDetails["Be"] = ["Beryllium", "Be", 9, 9.012183, 1.00]
            self.elemDetails["Bi"] = ["Bismuth", "Bi", 209, 208.980388, 1.00]
            self.elemDetails["B"] = ["Boron", "B", 11, 11.009305, 0.802]
            self.elemDetails["Br"] = ["Bromine", "Br", 79, 78.918336, 0.5069]
            self.elemDetails["Cd"] = ["Cadmium", "Cd", 114, 113.903361, 0.2873]
            self.elemDetails["Ca"] = ["Calcium", "Ca", 40, 39.962591, 0.9695]
            self.elemDetails["44Ca"] = [
                "Calcium",
                "Ca",
                44,
                43.955485,
                0.0208,
            ]  # 3.992894

            self.elemDetails["C"] = ["Carbon", "C", 12, 12.0, 0.9893]
            self.elemDetails["12C"] = ["Carbon", "C", 12, 12.0, 0.9893]
            self.elemDetails["13C"] = [
                "Carbon",
                "C",
                13,
                13.00335483507,
                0.0107,
            ]  # 1.00335

            self.elemDetails["Ce"] = ["Cerium", "Ce", 140, 139.905442, 0.8848]
            self.elemDetails["Cs"] = ["Cesium", "Cs", 133, 132.905433, 1.00]
            self.elemDetails["Cl"] = ["Chlorine", "Cl", 35, 34.968853, 0.7577]
            self.elemDetails["35Cl"] = ["Chlorine", "Cl", 35, 34.968853, 0.7577]
            self.elemDetails["37Cl"] = [
                "Chlorine",
                "Cl",
                37,
                36.965903,
                0.2423,
            ]  # 1.997077

            self.elemDetails["Cr"] = ["Chromium", "Cr", 52, 51.94051, 0.8379]
            self.elemDetails["50Cr"] = [
                "Chromium",
                "Cr",
                50,
                49.946046,
                0.0435,
            ]  # -1.994464
            self.elemDetails["53Cr"] = [
                "Chromium",
                "Cr",
                53,
                52.940651,
                0.095,
            ]  # 1.000141
            self.elemDetails["54Cr"] = [
                "Chromium",
                "Cr",
                54,
                53.938882,
                0.0236,
            ]  # 1.998372

            self.elemDetails["Co"] = ["Cobalt", "Co", 59, 58.933198, 1.00]
            self.elemDetails["Cu"] = ["Copper", "Cu", 63, 62.929599, 0.6917]
            self.elemDetails["65Cu"] = [
                "Copper",
                "Cu",
                65,
                64.927792,
                0.3083,
            ]  # 1.998193

            self.elemDetails["Dy"] = ["Dysprosium", "Dy", 164, 163.929183, 0.282]
            self.elemDetails["Er"] = ["Erbium", "Er", 166, 165.930305, 0.336]
            self.elemDetails["Eu"] = ["Europium", "Eu", 153, 152.921243, 0.522]
            self.elemDetails["F"] = ["Fluorine", "F", 19, 18.998403, 1.00]
            self.elemDetails["Gd"] = ["Gadolinium", "Gd", 158, 157.924111, 0.2484]
            self.elemDetails["Ga"] = ["Gallium", "Ga", 69, 68.925581, 0.601]
            self.elemDetails["Ge"] = ["Germanium", "Ge", 74, 73.921179, 0.365]
            self.elemDetails["Au"] = ["Gold", "Au", 197, 196.96656, 1.00]
            self.elemDetails["Hf"] = ["Hafnium", "Hf", 180, 179.946561, 0.352]
            self.elemDetails["He"] = ["Helium", "He", 4, 4.002603, 1.00]
            self.elemDetails["Ho"] = ["Holmium", "Ho", 165, 164.930332, 1.00]
            self.elemDetails["H"] = ["Hydrogen", "H", 1, 1.007825, 0.999]
            self.elemDetails["1H"] = ["Hydrogen", "H", 1, 1.007825, 0.999]
            self.elemDetails["D"] = [
                "Hydrogen",
                "H",
                2,
                2.014102,
                0.001,
            ]  ## ATTENTION May be wrong. Just used for Cambridge
            self.elemDetails["2H"] = [
                "Hydrogen",
                "H",
                2,
                2.014102,
                0.001,
            ]  ## ATTENTION May be wrong. Just used for Cambridge
            self.elemDetails["In"] = ["Indium", "In", 115, 114.903875, 0.957]
            self.elemDetails["I"] = ["Iodine", "I", 127, 126.904477, 1.00]
            self.elemDetails["Ir"] = ["Iridium", "Ir", 193, 192.962942, 0.627]
            self.elemDetails["Fe"] = ["Iron", "Fe", 56, 55.934939, 0.9172]
            self.elemDetails["56Fe"] = ["Iron", "Fe", 56, 55.934939, 0.9172]
            self.elemDetails["54Fe"] = ["Iron", "Fe", 54, 53.939612, 0.058]  # -1.995327
            self.elemDetails["57Fe"] = ["Iron", "Fe", 57, 56.935396, 0.022]  # 1.000457

            self.elemDetails["Kr"] = ["Krypton", "Kr", 84, 83.911506, 0.57]
            self.elemDetails["La"] = ["Lanthanum", "La", 139, 138.906355, 0.9991]
            self.elemDetails["Pb"] = ["Lead", "Pb", 208, 207.976641, 0.524]
            self.elemDetails["Li"] = ["Lithium", "Li", 7, 7.016005, 0.9258]
            self.elemDetails["Lu"] = ["Lutetium", "Lu", 175, 174.940785, 0.974]
            self.elemDetails["Mg"] = ["Magnesium", "Mg", 24, 23.985045, 0.789]
            self.elemDetails["25Mg"] = [
                "Magnesium",
                "Mg",
                25,
                24.985839,
                0.10,
            ]  # 1.000794
            self.elemDetails["26Mg"] = [
                "Magnesium",
                "Mg",
                26,
                25.982595,
                0.111,
            ]  # 1.99755

            self.elemDetails["Mn"] = ["Manganese", "Mn", 55, 54.938046, 1.00]
            self.elemDetails["Hg"] = ["Mercury", "Hg", 202, 201.970632, 0.2965]
            self.elemDetails["Mo"] = ["Molybdenum", "Mo", 98, 97.905405, 0.2413]
            self.elemDetails["Nd"] = ["Neodymium", "Nd", 142, 141.907731, 0.2713]
            self.elemDetails["Ne"] = ["Neon", "Ne", 20, 19.992439, 0.906]
            self.elemDetails["Ni"] = ["Nickel", "Ni", 58, 57.935347, 0.6827]
            self.elemDetails["Nb"] = ["Niobium", "Nb", 93, 92.906378, 1.00]
            self.elemDetails["N"] = ["Nitrogen", "N", 14, 14.003074, 0.9963]
            self.elemDetails["14N"] = ["Nitrogen", "N", 14, 14.003074, 0.9963]
            self.elemDetails["15N"] = ["Nitrogen", "N", 15, 15.0001088982, 0.00364]
            self.elemDetails["Os"] = ["Osmium", "Os", 192, 191.961487, 0.41]
            self.elemDetails["O"] = ["Oxygen", "O", 16, 15.994915, 0.9976]
            self.elemDetails["16O"] = ["Oxygen", "O", 16, 15.994915, 0.9976]
            self.elemDetails["18O"] = ["Oxygen", "O", 18, 17.99915961286, 0.00205]  # 2.004245
            self.elemDetails["Pd"] = ["Palladium", "Pd", 106, 105.903475, 0.2733]
            self.elemDetails["P"] = ["Phosphorus", "P", 31, 30.973763, 1.00]
            self.elemDetails["Pt"] = ["Platinum", "Pt", 195, 194.964785, 0.338]
            self.elemDetails["K"] = ["Potassium", "K", 39, 38.963708, 0.932]
            self.elemDetails["41K"] = [
                "Potassium",
                "K",
                41,
                40.961825,
                0.0673,
            ]  # 1.998117

            self.elemDetails["Pr"] = ["Praseodymium", "Pr", 141, 140.907657, 1.00]
            self.elemDetails["Re"] = ["Rhenium", "Re", 187, 186.955765, 0.626]
            self.elemDetails["Rh"] = ["Rhodium", "Rh", 103, 102.905503, 1.00]
            self.elemDetails["Rb"] = ["Rubidium", "Rb", 85, 84.9118, 0.7217]
            self.elemDetails["Ru"] = ["Ruthenium", "Ru", 102, 101.904348, 0.316]
            self.elemDetails["Sm"] = ["Samarium", "Sm", 152, 151.919741, 0.267]
            self.elemDetails["Sc"] = ["Scandium", "Sc", 45, 44.955914, 1.00]
            self.elemDetails["Se"] = ["Selenium", "Se", 80, 79.916521, 0.496]
            self.elemDetails["Si"] = ["Silicon", "Si", 28, 27.976928, 0.9223]
            self.elemDetails["Ag"] = ["Silver", "Ag", 107, 106.905095, 0.5184]
            self.elemDetails["Na"] = ["Sodium", "Na", 23, 22.98977, 1.00]
            self.elemDetails["Sr"] = ["Strontium", "Sr", 88, 87.905625, 0.8258]
            self.elemDetails["S"] = ["Sulfur", "S", 32, 31.972072, 0.9502]
            self.elemDetails["32S"] = ["Sulfur", "S", 32, 31.972072, 0.9502]
            self.elemDetails["34S"] = ["Sulfur", "S", 34, 33.967868, 0.0421]  # 1.995796

            self.elemDetails["Ta"] = ["Tantalum", "Ta", 181, 180.948014, 0.9999]
            self.elemDetails["Te"] = ["Tellurium", "Te", 130, 129.906229, 0.338]
            self.elemDetails["Tb"] = ["Terbium", "Tb", 159, 158.92535, 1.00]
            self.elemDetails["Tl"] = ["Thallium", "Tl", 205, 204.97441, 0.7048]
            self.elemDetails["Th"] = ["Thorium", "Th", 232, 232.038054, 1.00]
            self.elemDetails["Tm"] = ["Thulium", "Tm", 169, 168.934225, 1.00]
            self.elemDetails["Sn"] = ["Tin", "Sn", 120, 119.902199, 0.324]
            self.elemDetails["Ti"] = ["Titanium", "Ti", 48, 47.947947, 0.738]
            self.elemDetails["W"] = ["Tungsten", "W", 184, 183.950953, 0.3067]
            self.elemDetails["U"] = ["Uranium", "U", 238, 238.050786, 0.9927]
            self.elemDetails["V"] = ["Vanadium", "V", 51, 50.943963, 0.9975]
            self.elemDetails["Xe"] = ["Xenon", "Xe", 132, 131.904148, 0.269]
            self.elemDetails["Yb"] = ["Ytterbium", "Yb", 174, 173.938873, 0.318]
            self.elemDetails["Y"] = ["Yttrium", "Y", 89, 88.905856, 1.00]
            self.elemDetails["Zn"] = ["Zinc", "Zn", 64, 63.929145, 0.486]
            self.elemDetails["66Zn"] = ["Zinc", "Zn", 66, 65.926035, 0.279]  # 1.99689
            self.elemDetails["67Zn"] = ["Zinc", "Zn", 67, 66.927129, 0.041]  # 2.997984
            self.elemDetails["68Zn"] = ["Zinc", "Zn", 68, 67.924846, 0.188]  # 3.995701

            self.elemDetails["Zr"] = ["Zirconium", "Zr", 90, 89.904708, 0.5145]

        else:
            self.elemDetails = elemDetails

    # INTERNAL METHOD used for parsing
    # parses a number
    def _parseNumber(self, formula, pos):
        if pos >= len(formula):
            return -1, 1

        if formula[pos].isdigit():
            num = ""
            while formula[pos].isdigit() and pos < len(formula):
                num = num + formula[pos]
                pos = pos + 1
            return pos, int(num)
        else:
            return pos, 1

    # INTERNAL METHOD used for parsing
    # parses an element
    def _parseStruct(self, formula, pos):
        elemDict = {}

        if formula[pos] == "(":
            pos = pos + 1
            while formula[pos] != ")":
                pos, elem = self._parseStruct(formula, pos)
                for kE in elem.keys():
                    if kE in elemDict.keys():
                        elemDict[kE] = elemDict[kE] + elem[kE]
                    else:
                        elemDict[kE] = elem[kE]
            pos, numb = self._parseNumber(formula, pos + 1)
            for kE in elemDict.keys():
                elemDict[kE] = elemDict[kE] * numb
            return pos, elemDict
        elif formula[pos] == "[":
            pos = pos + 1

            num = ""
            while formula[pos].isdigit() and pos < len(formula):
                num = num + formula[pos]
                pos = pos + 1
            if pos == "":
                raise Exception("Isotope description wrong")

            curElem = formula[pos]
            if (pos + 1) < len(formula) and formula[pos + 1].isalpha() and formula[pos + 1].islower():
                curElem = formula[pos : (pos + 2)]
            if curElem != "":
                pos = pos + len(curElem)
            else:
                raise Exception("Unrecognized element")

            if formula[pos] != "]":
                raise Exception("Malformed isotope: " + formula)
            pos = pos + 1

            pos, numb = self._parseNumber(formula, pos)
            elemDict[curElem] = numb
            elemDict[num + curElem] = numb

            return pos, elemDict

        else:
            curElem = formula[pos]
            if (pos + 1) < len(formula) and formula[pos + 1].isalpha() and formula[pos + 1].islower():
                curElem = formula[pos : (pos + 2)]
            if curElem != "":
                pos, numb = self._parseNumber(formula, pos + len(curElem))
                elemDict[curElem] = numb
                return pos, elemDict
            else:
                raise Exception("Unrecognized element")
        return -1

    # parses a formula into an element-dictionary (charge notation is stripped and ignored)
    def parseFormula(self, formula):
        clean, _ = _parse_charge_suffix(formula.replace(" ", ""))
        return self._parseStruct("(" + clean + ")", 0)[1]

    def parseFormulaWithCharge(self, formula):
        """Parse a chemical formula and return (elemDict, charge).

        Supports charge notation like C15H20O6+, C15H20O6+2, C15H20O6++.
        Returns (element_dict, charge) where charge is an integer (0 for neutral).
        """
        clean, charge = _parse_charge_suffix(formula.replace(" ", ""))
        elems = self._parseStruct("(" + clean + ")", 0)[1]
        return elems, charge

    def get_formal_charge(self, formula):
        """Return the formal charge of a chemical formula as an integer.

        Examples:
            'C15H20O6'   -> 0
            'C15H20O6+'  -> 1
            'C15H20O6+2' -> 2
            'C15H20O6++' -> 2
            'C15H20O6-'  -> -1
            'C15H20O6-2' -> -2
        """
        _, charge = _parse_charge_suffix(formula.replace(" ", ""))
        return charge

    # method determines if a given element represent an isotope other the the main isotope of a given element
    # e.g. isIso("13C"): True; isIso("12C"): False
    def isIso(self, iso):
        return not (iso[0].isalpha())

    # returns the element for a given isotope
    def getElementFor(self, elem):
        if not (self.isIso(elem)):
            raise Exception("Element was provided. Isotope is required")
        else:
            num = ""
            pos = 0
            while elem[pos].isdigit() and pos < len(elem):
                num = num + elem[pos]
                pos = pos + 1
            if pos == "":
                raise Exception("Isotope description wrong")
            curElem = elem[pos]

            if (pos + 1) < len(elem) and elem[pos + 1].isalpha() and elem[pos + 1].islower():
                curElem = elem[pos : (pos + 2)]
            if curElem != "":
                pos = pos + len(curElem)
            else:
                raise Exception("Unrecognized element")

            return curElem, num

    # helper method: n over k
    def noverk(self, n, k):
        from functools import reduce

        return reduce(lambda a, b: a * (n - b) // (b + 1), range(k), 1)

    # helper method: calculates the isotopic ratio
    def getIsotopologueRatio(self, c, s, p):
        return pow(p, s) * self.noverk(c, s)

    def getMassOffset(self, elems):
        fElems = {}
        fIso = {}
        ret = 0
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if curElem not in fIso:
                    fIso[curElem] = []
                fIso[curElem].append((iso, elems[elem]))

        for elem in fElems:
            rem = 0
            if elem in fIso:
                for x in fIso[elem]:
                    rem = rem + x[1]
            p = self.elemDetails[elem][4]
            c = fElems[elem] - rem
            ret = ret * pow(p, c)
        for iso in fIso:
            for cIso in fIso[iso]:
                ret = ret + (self.elemDetails[str(cIso[0] + iso)][3] - self.elemDetails[iso][3]) * cIso[1]
        return ret

    def getAbundance(self, elems):
        fElems = {}
        fIso = {}
        ret = 1.0
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if curElem not in fIso:
                    fIso[curElem] = []
                fIso[curElem].append((iso, elems[elem]))
        for elem in fElems:
            rem = 0
            if elem in fIso:
                for x in fIso[elem]:
                    rem = rem + x[1]
            p = self.elemDetails[elem][4]
            c = fElems[elem] - rem
            ret = ret * pow(p, c)
        for iso in fIso:
            for cIso in fIso[iso]:
                ret = ret * self.getIsotopologueRatio(fElems[iso], cIso[1], self.elemDetails[str(cIso[0]) + iso][4])
        return ret

    def getAbundanceToMonoisotopic(self, elems):
        onlyElems = {}
        for elem in elems:
            if not (self.isIso(elem)):
                onlyElems[elem] = elems[elem]

        return self.getAbundance(elems) / self.getAbundance(onlyElems)

    # calculates the molecular weight of a given elemental collection (e.g. result of parseFormula)
    # If charge is non-zero, the mass is adjusted for the gain/loss of electrons:
    #   positive charge (cation): subtract charge * ELECTRON_MASS
    #   negative charge (anion):  add abs(charge) * ELECTRON_MASS
    def calcMolWeight(self, elems, charge=0):
        mw = 0.0
        for elem in elems.keys():
            if not (self.isIso(elem)):
                mw = mw + self.elemDetails[elem][3] * elems[elem]
            else:
                curElem, iso = self.getElementFor(elem)
                mw = mw + self.elemDetails[iso + curElem][3] * elems[elem] - self.elemDetails[curElem][3] * elems[elem]

        if charge != 0:
            mw = mw - charge * ELECTRON_MASS

        return mw

    # returns putaive isotopes for a given mz difference
    def getPutativeIsotopes(self, mzdiff, atMZ, z=1, ppm=5.0, maxIsoCombinations=1, used=[]):
        mzdiff = mzdiff * z
        maxIsoCombinations = maxIsoCombinations - 1

        ret = []

        for elem in self.elemDetails:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)
                diff = self.elemDetails[iso + curElem][3] - self.elemDetails[curElem][3]

                if maxIsoCombinations == 0:
                    if abs(mzdiff - diff) < (atMZ * ppm / 1000000.0):
                        x = [y for y in used]
                        x.append(iso + curElem)

                        d = {}
                        for y in x:
                            if y not in d:
                                d[y] = 0
                            d[y] = d[y] + 1
                        ret.append(d)

                else:
                    # print("next level with", mzdiff-diff, "after", iso+elem, self.elemDetails[iso+elem][3],self.elemDetails[elem][3])
                    x = [y for y in used]
                    x.append(iso + curElem)
                    x = self.getPutativeIsotopes(
                        mzdiff - diff,
                        atMZ=atMZ,
                        z=1,
                        ppm=ppm,
                        maxIsoCombinations=maxIsoCombinations,
                        used=x,
                    )
                    ret.extend(x)

        if maxIsoCombinations > 0:
            x = [y for y in used]
            x = self.getPutativeIsotopes(
                mzdiff,
                atMZ=atMZ,
                z=1,
                ppm=ppm,
                maxIsoCombinations=maxIsoCombinations,
                used=x,
            )
            ret.extend(x)

        return ret

    # prints a given elemental collection in form of a sum formula
    def flatToString(self, elems, prettyPrintWithHTMLTags=False, subStart="<sub>", subEnd="</sub>"):
        if isinstance(elems, str):
            elems = self.parseFormula(elems)

        if not prettyPrintWithHTMLTags:
            subStart = ""
            subEnd = ""

        fElems = {}
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if curElem in fElems:
                    fElems[curElem] = fElems[curElem] - elems[elem]
                    fElems["[" + iso + curElem + "]"] = elems[elem]
                else:
                    fElems["[" + iso + curElem + "]"] = elems[elem]

        return "".join([("%s%s%d%s" % (e, subStart, fElems[e], subEnd) if fElems[e] > 1 else "%s" % e) for e in sorted(fElems.keys())])

    def getIsotopes(self, minInt=0.02):
        ret = {}
        for elem in self.elemDetails:
            if self.isIso(elem):
                el = self.getElementFor(elem)
                prob = self.elemDetails[elem][4] / self.elemDetails[el[0]][4]
                if prob >= minInt:
                    ret[elem] = (
                        self.elemDetails[elem][3] - self.elemDetails[el[0]][3],
                        prob,
                    )

        return ret

    def calcDifferenceBetweenElemDicts(self, elemsFragment, elemsParent):
        loss = {}
        for elem in elemsParent:
            l = 0
            if elem in elemsFragment:
                l = elemsParent[elem] - elemsFragment[elem]
            else:
                l = elemsParent[elem]
            if l > 0:
                loss[elem] = l
        return loss

    def calcDifferenceBetweenSumFormulas(self, sfFragment, sfParent):
        return self.calcDifferenceBetweenElemDicts(self.parseFormula(sfFragment), self.parseFormula(sfParent))


# helper method that returns the mass of a given isotope. Used in the main interface of MetExtract II
def getIsotopeMass(isotope):
    fT = formulaTools()
    mass = -1
    element = ""
    for i in fT.elemDetails:
        elemDetails = fT.elemDetails[i]
        if elemDetails[1][0].isdigit():
            if isotope == elemDetails[1]:
                mass = elemDetails[3]
                element = elemDetails[0]
        elif elemDetails[1][0].isalpha():
            if isotope == "%d%s" % (elemDetails[2], elemDetails[1]):
                mass = elemDetails[3]
                element = elemDetails[0]
    return mass, element


def getElementOfIsotope(isotope):
    fT = formulaTools()
    return fT.elemDetails[isotope][1]


def choose(n, r):
    from math import factorial

    return factorial(n) // factorial(r) // factorial(n - r)


def calcIsoEnrichment(a, s, r):
    """Calculate the fractional isotopic enrichment from a pair of isotopolog
    intensities separated by s labelling positions.

    a: total number of labelling positions (Xn)
    s: number of labelling positions between the two used isotopologs
    r: intensity ratio of the more distant isotopolog relative to the reference isotopolog
    """
    return choose(a, s) ** (1.0 / s) / (choose(a, s) ** (1.0 / s) + r ** (1.0 / s))


def isotopologLabel(x, xCount):
    """Name an isotopolog at labelling position x (out of xCount total positions) using
    the combined native/labelled notation "M+x / M'-(Xn-x)"."""
    y = xCount - x
    native = f"M+{x}" if x >= 0 else f"M{x}"
    labeled = f"M'-{y}" if y >= 0 else f"M'+{-y}"
    return f"{native} / {labeled}"


if __name__ == "__main__":
    fT = formulaTools()

    formulas = ["C9H11NO2"]

    res = []
    for form in formulas:
        elems = fT.parseFormula(form)
        res.append(
            [
                form,
                str(elems),
                "%.5f" % fT.calcMolWeight(elems),
                "%.5f" % (fT.calcMolWeight(elems) + 1.007276),
                "%.5f" % (fT.calcMolWeight(elems) + 18.033823),
                "%.5f" % (fT.calcMolWeight(elems) + 22.989218),
                "%.5f" % (fT.calcMolWeight(elems) + 44.998201),
                "%.5f" % (fT.calcMolWeight(elems) - 1.007276),
                fT.flatToString(elems),
            ]
        )

    from .utils import printAsTable

    printAsTable(
        [
            "Input",
            "Parsed",
            "Mass",
            "[M+H]+",
            "[M+NH4]+",
            "[M+Na]+",
            "[M+Fa-H]-",
            "[M-H]-",
            "Rendered",
        ],
        res,
        printInExcelFormat=True,
    )

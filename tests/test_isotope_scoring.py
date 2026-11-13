"""Tests for src/isotopeScoring.py."""

import json

from src.isotopeScoring import ISOTOPOLOG_NAMES, ISOTOPOLOGS, UNDEFINED_SCORE, compute_isotope_pattern, theoretical_isotopolog_ratio
from src.formulaTools import formulaTools


def _isotopologs_json(**medians):
    return json.dumps({name: {"median": value} for name, value in medians.items()})


class TestTheoreticalIsotopologRatio:
    def test_missing_element_is_zero(self):
        ft = formulaTools()
        elems = ft.parseFormula("C6H12O6")
        assert theoretical_isotopolog_ratio(elems, "S", "34S", 1) == 0.0

    def test_present_element_is_positive(self):
        ft = formulaTools()
        elems = ft.parseFormula("C6H12O6")
        ratio = theoretical_isotopolog_ratio(elems, "C", "13C", 1)
        assert ratio > 0.0


class TestComputeIsotopePattern:
    def test_perfect_match_is_zero(self):
        ft = formulaTools()
        formula = "C6H12O6"
        elems = ft.parseFormula(formula)
        medians = {}
        for name, element_symbol, isotope_key, count in ISOTOPOLOGS:
            medians[name] = theoretical_isotopolog_ratio(elems, element_symbol, isotope_key, count)
        score, ratios = compute_isotope_pattern(formula, _isotopologs_json(**medians))
        assert score == 0.0 or abs(score) < 1e-12
        assert set(ratios.keys()) == set(ISOTOPOLOG_NAMES)
        for name in ISOTOPOLOG_NAMES:
            assert abs(ratios[name]["measured"] - ratios[name]["theoretical"]) < 1e-12

    def test_all_zero_observed_gives_positive_score_for_carbon_containing_formula(self):
        score, ratios = compute_isotope_pattern("C6H12O6", _isotopologs_json(**{"M+13C": 0, "M+13C2": 0, "M+15N": 0, "M+34S": 0, "M+37Cl": 0, "M-54Fe": 0}))
        assert score > 0.0
        assert ratios["M+13C"]["measured"] == 0.0
        assert ratios["M+13C"]["theoretical"] > 0.0

    def test_score_is_sum_of_absolute_differences_not_mean(self):
        # M+15N and M+34S are each off by 0.1; all other isotopologs match their theoretical
        # value exactly (measured == theoretical) -> score should be the SUM (0.2), not the
        # mean (0.2 / 6).
        ft = formulaTools()
        formula = "C6H12O6"
        elems = ft.parseFormula(formula)
        medians = {}
        for name, element_symbol, isotope_key, count in ISOTOPOLOGS:
            medians[name] = theoretical_isotopolog_ratio(elems, element_symbol, isotope_key, count)
        medians["M+15N"] += 0.1
        medians["M+34S"] += 0.1
        score, _ratios = compute_isotope_pattern(formula, _isotopologs_json(**medians))
        assert abs(score - 0.2) < 1e-9

    def test_missing_other_isotopologs_is_undefined(self):
        score, ratios = compute_isotope_pattern("C6H12O6", None)
        assert score == UNDEFINED_SCORE
        assert ratios == {}
        score, ratios = compute_isotope_pattern("C6H12O6", "")
        assert score == UNDEFINED_SCORE
        assert ratios == {}

    def test_missing_formula_is_undefined(self):
        assert compute_isotope_pattern(None, _isotopologs_json(**{"M+13C": 0.01}))[0] == UNDEFINED_SCORE
        assert compute_isotope_pattern("", _isotopologs_json(**{"M+13C": 0.01}))[0] == UNDEFINED_SCORE

    def test_unparseable_formula_is_undefined(self):
        assert compute_isotope_pattern("[13C", _isotopologs_json(**{"M+13C": 0.01}))[0] == UNDEFINED_SCORE

    def test_unrelated_isotopologs_are_ignored(self):
        # M-13C / Mp+* are not part of the compared set; they must not affect the score.
        score_without, _ = compute_isotope_pattern("C6H12O6", _isotopologs_json(**{"M+13C": 0.0}))
        score_with_extra, _ = compute_isotope_pattern(
            "C6H12O6",
            json.dumps(
                {
                    "M+13C": {"median": 0.0},
                    "M-13C": {"median": 0.5},
                    "Mp+D": {"median": 0.9},
                }
            ),
        )
        assert score_without == score_with_extra

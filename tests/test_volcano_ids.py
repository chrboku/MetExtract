import pandas as pd

from src.mePyGuis.statisticsModule import UnivariateAnalysis


def test_calculate_volcano_data_preserves_string_num_and_ogroup_ids():
    feature_id = "feature-A"
    data = pd.DataFrame(
        {
            "group1-a": [10.0],
            "group1-b": [12.0],
            "group2-a": [5.0],
            "group2-b": [6.0],
        },
        index=[feature_id],
    )
    metadata = pd.DataFrame(
        {
            "num": {feature_id: feature_id},
            "ogroup": {feature_id: "metabolite-A"},
        }
    )

    result = UnivariateAnalysis.calculate_volcano_data(
        data,
        ["group1-a", "group1-b"],
        ["group2-a", "group2-b"],
        metadata=metadata,
    )

    assert result["success"] is True
    assert result["featurePairIDs"] == [feature_id]
    assert result["featureGroupIDs"] == ["metabolite-A"]

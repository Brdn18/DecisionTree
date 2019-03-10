import pytest

from sklearn.utils.estimator_checks import check_estimator

from decision_tree_c45 import C45Classifier


@pytest.mark.parametrize(
    "Estimator", [C45Classifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

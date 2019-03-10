import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from decision_tree_c45 import C45Classifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_classifier(data):
    X, y = data
    clf = C45Classifier()
    
    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    clf.printTree()

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)

    

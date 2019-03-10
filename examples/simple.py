import numpy as np

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append('.')
sys.path.append('..')
from decision_tree_c45 import C45Classifier


def printDecisionTree(estimator):

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
        "the following tree structure:"
        % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                "node %s."
                % (node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                    ))
    print()

def main():
    clf = C45Classifier()
    X, y = load_iris(return_X_y=True)
    #print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf.fit(X_train, y_train, force_nominal_indices=[], feature_names=['1st', '2nd', '3rd', '4th'])
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')


    clf.printTree()

    print(X_test.shape)
    print( y_test)
    print('-------------')
    y_pred = clf.predict(X_test)

    print(clf.feature_importances())
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('-------------')

    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(X_train, y_train)

    printDecisionTree(tree)

    y_pred = tree.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))



if __name__ == "__main__":
    main()
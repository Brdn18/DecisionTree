import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
import math

class Node:
    def __init__(self, 
                depth,
                label = None,
                description = None,
                best_feature_id = -1,
                best_feature_threshold = -1 * float('inf'),
                impurity = 0):
        self.depth = depth
        self.label = label
        self.description = description
        self.best_feature_id = best_feature_id
        self.best_feature_threshold = best_feature_threshold
        self.impurity = impurity
        self.counts = {}
        self.probabilities = {}

        self.children = {}



class C45Classifier(BaseEstimator, ClassifierMixin):
    """ Classifier which implements c45 algorithm

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    algo : str, default='c45'
        A parameter used for selecting proper tree algorithm. Now only two supported - 'id3' and 'c45'
        The only difference between them is the eway how amount of additional info is measured - via information gain or gain ratio
    max_depth: int, default=10
        Max depth of the tree
    min_samples_split: int, default=2
        Min size of a node for further split

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, 
                algo='c45',
                max_depth=10,
                min_samples_split=2):
        self.algo = algo
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        

    def fit(self, X, y, 
            types_auto=True,
            force_nominal_indices=[],
            force_continuous_indices=[],
            feature_names = [],
            class_names = []):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        types_auto: boolean, default=True
            If true, try to determine if each variable is categorical or numeric; if False, all categories are considered as categorical
        force_nominal_indices: array-like
            Any variable can be explicitly marked as categorical. 
            If both parameters, force_nominal_indices and force_continuous_indices are set and have intersections, parameter force_nominal_indices has priority
        force_continuous_indices: array-like
            Any variable can be explicitly marked as numeric/continuous. 
            If both parameters, force_nominal_indices and force_continuous_indices are set and have intersections, parameter force_nominal_indices has priority
        feature_names: array_like, shape(n_features,)
            Names of features; set to their numbers if no names are provided

        Returns
        -------
        self : object
            Returns self.
        """
        self.nominal_features_ = []
        self.feature_names_ = []
        self.attrs_for_indices = {}
        


        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = np.array(unique_labels(y))

        self.X_ = np.array(X)
        self.y_ = np.array(y).reshape(-1, 1)

        

        self.data_ = np.append(self.X_, self.y_, axis=1)

        #print('shapes', self.X_.shape, self.y_.shape, self.data_.shape)
        
        #self.data_ = np.hstack( (self.X_, self.y_) )
        #print('Joined data with classes', self.data_)

        self.total_attributes_no = self.X_.shape[1]
        if len(feature_names) > 0:
            if len(feature_names) != self.total_attributes_no:
                raise ValueError("Attribute labels shape doesnot fit data shape")
        else:
            feature_names = [i for i in range(self.total_attributes_no)]

        self.feature_names_ = feature_names
        self.attr_indices_ = [i for i in range(self.total_attributes_no)]
        #print('total attrs', self.total_attributes_no, self.feature_names_)
        
        
        # Infer types of columns
        self.nominal_features_ = np.full(self.total_attributes_no, True)
        if types_auto == True:
            for i in range(self.total_attributes_no):
                self.nominal_features_[i] = self._is_attr_discrete(i)
        
        for cont_i in force_continuous_indices:
            self.nominal_features_[cont_i] = False
        for nom_i in force_nominal_indices:
            self.nominal_features_[nom_i] = True

        for i in range(self.total_attributes_no):
            self.attrs_for_indices[i] = np.unique(X[:,i])

        # 
        #build tree
        self._generate_tree()


        return self

    def predict(self, X):
        """ An implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        dummy = np.zeros( X.shape[0] )

        default_class_threshold = 0.5
        index = 0
        for row in X:
            probas = self._predict_proba_one(row)
            if probas is None:
                continue
            for c, p in probas.items():
                if p >= default_class_threshold:
                    dummy[index] = c
                    break
            #print(row, probas, index, dummy[index])    
            index += 1

        return dummy

    def predict_proba(self, data):
        """ An implementation of a probabilitis prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            Probability for each class for each incoming sample.
        """
        output = []
        for row in data:
            proba = self._predict_proba_one(row)
            o = []
            for _, v in proba.items():
                o.append(v)
            output.append(o)

        return np.array(output)

    def _predict_proba_one(self, row):
        #traverse tree
        node = self.tree
        
        if node is None:
            return None

        while True:
            feature_id = node.best_feature_id
            observed_feature_value = row[feature_id]

            next_node = None

            if self.nominal_features_[feature_id] == True:
                for link, child in node.children.items():
                    if observed_feature_value == link:
                        next_node = child
                        break
            else:
                threshold = node.best_feature_threshold
                if len(node.children) == 2:
                    if observed_feature_value <= threshold:
                        next_node = node.children['left']
                    else:
                        next_node = node.children['right']
            
            if next_node == None:
                break
            
            node = next_node

        probas = node.probabilities
        return probas
     
    def feature_importances(self):
        node = self.tree
        importances = {}

        node_0_size = 0
        for _, v in node.counts.items():
            node_0_size += v

        node_layers, total_nodes = self._nodes_to_array()

        for layer in node_layers:
            for node in layer:
                feature_id = node.best_feature_id
                if feature_id == -1:
                    continue
                impurity = node.impurity

                total_entries = 0
                for _, v in node.counts.items():
                    total_entries += v

                children_importances = 0
                #check children
                for _, child in node.children.items():
                    child_impurity = child.impurity
                    if child_impurity == -1*float('inf'):
                        continue 
                    
                    total_child_entries = 0
                    for _, v in child.counts.items():
                        total_child_entries += v
                    children_importances += child_impurity * total_child_entries
                #print(impurity)
                node_importance = impurity * total_entries - children_importances
                node_importance /= node_0_size
                #print('Importance for', self.feature_names_[feature_id], 'is', node_importance, '(total', total_entries, ', node impurity', impurity, 'children cumulative importances', children_importances, ')')
                #print(feature_id)
                if feature_id in importances:
                    importances[feature_id] = max(node_importance, importances[feature_id])
                else:
                    importances[feature_id] = node_importance

        #print(importances)
        return importances


    def _nodes_to_array(self):
        layers = []
    
        layer_id = 0
        node = self.tree
        layer = []
        layer.append(node)
        layers.append(layer)

        total_nodes = len(layer)

        while True:
            all_nodes_in_layer = layers[layer_id]
            print('Layer size', len(all_nodes_in_layer))
            new_layer = []
            for n in all_nodes_in_layer:
                for _, child in n.children.items():
                    new_layer.append(child)
            if len(new_layer) == 0:
                break

            layer_id += 1
            layers.append(new_layer)
            total_nodes += len(new_layer)

        return layers, total_nodes




    def _log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x,2)


    def _entropy(self, data):
        s = len(data)
        if s == 0:
            return 0
        
        num_classes = np.array([0 for u in self.classes_])

        for row in data:
            class_index = np.where(self.classes_ == row[-1])
            num_classes[class_index] += 1
            
        num_classes = [x/s for x in num_classes]
        ent = 0
        for num in num_classes:
            ent += num * self._log(num)
        return (-1) * ent


    def _information_gain(self, data, subsets):
        #input : data and disjoint subsets of it
        #output : information gain
        s = len(data)
        #calculate impurity before split
        impurity_before_split = self._entropy(data)
        #calculate impurity after split
        weights = [len(subset)/s for subset in subsets]
        impurity_after_split = 0
        intrinsic_value = 0
        for i in range(len(subsets)):
            impurity_after_split += weights[i] * self._entropy(subsets[i])
            intrinsic_value += weights[i] * self._log(weights[i])
        #calculate total gain
        total_gain = impurity_before_split - impurity_after_split
        
        if self.algo == 'c45':
            gain_ratio = total_gain / ( -1 * intrinsic_value)
            return gain_ratio 
        elif self.algo == 'id3':
            return total_gain
        else:
            raise ValueError('Unsupported algo:' + self.algo)


    def _is_attr_discrete(self, attr_id):
        dtype = self.X_[:,attr_id].dtype
        if (dtype == np.float) or (dtype == np.float64) :
            return False
        return True

    def _ig_for_nominal_feature(self, data, feature_id):
        X_column = data[:, feature_id]

        unique_values_for_feature = np.unique(X_column)
        X_column_splits = [ [] for i in range(len(unique_values_for_feature)) ]
        
        X_samples_count = data.shape[0]
        for sample_id in range(X_samples_count):
            sample = data[sample_id]
            for unique_value_id in range(len(unique_values_for_feature)):
                unique_value = unique_values_for_feature[unique_value_id]
                if unique_value == X_column[sample_id]:
                    X_column_splits[unique_value_id].append(sample)
        
        ig = self._information_gain(data, X_column_splits)
        threshold = None
        return (ig, threshold, X_column_splits, unique_values_for_feature) #to return same type as for cont variable


    def _ig_for_cont_feature(self, data, feature_id):
        X_column_sorted = np.sort(data[:, feature_id])
        #X_column_sorted = data[ data[:,feature_id].argsort() ]

        feature_threshold = -1* float('inf')
        best_ig = -1*float('inf')

        X_column_splits = None
        unique_feature_values = []

        for j in range(len(X_column_sorted) - 1):
            if X_column_sorted[j] != X_column_sorted[j + 1]:
                threshold = (X_column_sorted[j] + X_column_sorted[j + 1]) / 2
                less = []
                greater = []
                for row in data:
                    if row[feature_id] > threshold:
                        greater.append(row)
                    else:
                        less.append(row)

                ig = self._information_gain(data, [less, greater])
                
                if ig > best_ig:
                    best_ig = ig
                    feature_threshold = threshold
                    X_column_splits = [less, greater]
                    unique_feature_values = ['<=' + str(threshold), '>' + str(threshold)]

        return (best_ig, feature_threshold, X_column_splits, unique_feature_values)


    def _calculate_classes(self, y):
        counts = {}
        for x in self.classes_:
            counts[x] = 0

        for value in y:
            counts[value] += 1
        
        return counts


        
    def _generate_tree(self):
        self.tree = self._recursive_generate_tree(data = self.data_, 
                                                    feature_ids=self.attr_indices_,
                                                    level = 0, label = 'root', verbose=1)


    def _recursive_generate_tree(self, data, feature_ids, level, label, verbose):
        #strategy
        #1. Check level. If exceeds max_depth - exit
        #2. Check amount of data. If less than min_split or min_leaf - exit
        #3. Cycle through all features. 
        #   3.1 If feature is continuous or ordinal - find best binary split
        #   3.2 If feature is nominal - split by categories. Order by categories frequencies and select max_split_categories;
        #       other categories join to 'other'
        #4. Select feature which produces best split. Create subsets of data according to the best split. 
        #5. For each subset: 
        #   - increase depth
        #6. Recursive call 

        node = Node(level, label)

        offset = '\t'*level
        if verbose > 0:
            print(offset + 'Level', level, '; label', label)
            print(offset + 'Incoming data shape:', data.shape)
            print(offset + 'Incoming feature_ids:', feature_ids)

        if len(data) == 0:
            if verbose > 0:
                print(offset + 'Not enough data at all, terminate')
            node.description = 'Not enough data at all, terminate'
            return None

        node.counts = self._calculate_classes(data[:, -1])
        if verbose > 0:
            print(offset + 'Node counts',node.counts)
        for c, f in node.counts.items():
            node.probabilities[c] = f / len(data[:, -1])

        if len(data) <= self.min_samples_split:
            if verbose > 0:
                print(offset + 'Not enough data to go deeper, terminate')
            node.description = 'Not enough data to go deeper, terminate'
            return None

        if level > self.max_depth:
            if verbose > 0:
                print(offset + 'Max depth exceeded, terminate')
            node.description = 'Max depth exceeded, terminate'
            return None

        if self._check_one_class_remains(data):
            if verbose > 0:
                print(offset + 'One class remains, terminate')
            node.description = 'One class remains, terminate'
            return node


        new_level = level + 1

        best_feature_id = -1
        max_ig = -1 * float('inf')
        best_threshold = None
        data_splits = []
        description = None
        unique_feature_values = []

        for feature_id in feature_ids:
            ig = max_ig
            threshold = best_threshold
            splits = None
            unique_values = []
            if self.nominal_features_[feature_id] == True:
                #nominal feature
                (ig, threshold, splits, unique_values) = self._ig_for_nominal_feature(data, feature_id)
                description = 'nominal'
            else:
                #contunuous or ordinal feature
                (ig, threshold, splits, unique_values) = self._ig_for_cont_feature(data, feature_id)
                description = 'non-nominal'
            if ig > max_ig:    
                best_feature_id = feature_id
                best_threshold = threshold
                max_ig = ig
                data_splits = splits
                unique_feature_values = unique_values
                if verbose > 0:
                    print(offset + 'Feature', self.feature_names_[feature_id], 'has better split')
            else:
                if verbose > 0:
                    print(offset + 'Feature', self.feature_names_[feature_id], 'has less IG, ignoring')
        

        node.best_feature_id = best_feature_id
        node.best_feature_threshold = best_threshold
        node.description = description
        node.impurity = max_ig

        if best_feature_id < 0:
            if verbose > 0:
                print(offset + 'No better split. Leaf node')
            return node

        if data_splits is None:
            if verbose > 0:
                print(offset + 'No splits found. Leaf node')
            return node
    
        if len(data_splits) != len(unique_feature_values):
            if verbose > 0:
                print(offset + 'sizes (splits, values):', len(data_splits), len(unique_feature_values))
                print(offset + 'Shapes between splits and unique features don\'t match. Leaf node?')
            return node
    


        new_feature_ids = feature_ids.copy()
        if best_feature_id in new_feature_ids:
            new_feature_ids.remove(best_feature_id)
            
        if self.nominal_features_[node.best_feature_id] == True:
            i = 0
            for split in data_splits:
                if verbose > 0:
                    print(offset + 'Data shape for child:', np.array(split).shape)
                op = ''
                if self.nominal_features_[best_feature_id] == True:
                    op = ' is '
                unique_value = unique_feature_values[i]
                new_label = str(self.feature_names_[best_feature_id]) + op + str(unique_value)    
                
                child = self._recursive_generate_tree(np.array(split), new_feature_ids, new_level, new_label, verbose)
                if child != None:
                    #Check if our child split improves our knowledge; if probs remain the same, child is useless
                    if node.counts != child.counts:
                        node.children[unique_value] = child
                i += 1
        else:
            #Expected data splits should be 2

            for i in range(2):
                unique_value = unique_feature_values[i]
                new_label = str(self.feature_names_[best_feature_id]) + str(unique_value)
                child = self._recursive_generate_tree(np.array(data_splits[i]), 
                                                        new_feature_ids, new_level, new_label, verbose)
                if child != None:
                    if i == 0:
                        node.children['left'] = child
                    else:
                        node.children['right'] = child
        

        return node




    def _check_one_class_remains(self, data):
        uniques = np.unique(data[:,-1])
        if len(uniques) <= 1:
            return True

        return False

    def _get_major_class_index(self, data):
        #print('classes', self.classes_)
        freq = [0]*len(self.classes_)
        #print('classes', self.classes_)
        
        freq = np.array(freq)
        for row in data:
            #print('row-1, row:', row[-1], row)
            index = np.where(self.classes_ == row[-1])
            freq[index] += 1
        max_ind = np.where(freq == max(freq))

        #print('freqs', freq)
        return max_ind

    def _get_class_frequencies(self, data):
        #print('classes', self.classes_)
        freqs = [0]*len(self.classes_)
        #print('classes', self.classes_)
        
        freqs = np.array(freqs)
        for row in data:
            #print('row-1, row:', row[-1], row)
            index = np.where(self.classes_ == row[-1])
            freqs[index] += 1
        #print('freqs', freq)
        return freqs

    def printTree(self):
        print('--- Tree ---')
        self.printNode(self.tree)

    def printNode(self, node, indent=""):
        
        print(indent + 'Label:', node.label)
        print(indent + 'Description:', node.description)
        
        total_entries = 0
        classes_count_str = 'Classes count: ['
        for c, p in node.counts.items():
            classes_count_str += str(c) + ':' + str(p) + '; '
            total_entries += p
        classes_count_str += ']'
        
        classes_prob_str = 'Classes probs: ['
        for c, p in node.probabilities.items():
            classes_prob_str += str(c) + ':' + str(p) + '; '
#            classes_prob_str += str(self.feature_names_[c]) + ':' + str(p / total_entries) + ';'
        classes_prob_str += ']'

        print(indent + classes_count_str)
        print(indent + classes_prob_str)
        print(indent + 'Best feature to split:', self.feature_names_[node.best_feature_id])
        print(indent + 'Best feature threshold:', node.best_feature_threshold)


        for link, child in node.children.items():
            print(indent+'Link to child', link)
            self.printNode(child, indent + '\t')




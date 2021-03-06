# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    vals = np.unique(x)

    d = {v: [] for v in vals}

    for index, val in enumerate(x):
        
        d[val].append(index)

    return d


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    h = 0
    vals = np.unique(y)
    for i in vals:
        
        h = h + p(y,i) * np.log2(p(y,i))
    
    h = h * (-1)

    return (h)


def p(z,val):
    
    return float((z==val).sum()/len(z))

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    h_y = entropy(y)

    h_yx = 0

    unique_x = np.unique(x)
    dict_x = partition(x)

    for key in dict_x.keys():
        prob = float(len(dict_x[key])/len(x))
        yx = [y[i] for i in dict_x[key]]
        h_yx_i = entropy(yx)
        h_yx = h_yx + prob*h_yx_i 

    i_yx = h_y - h_yx

    return(i_yx)       

def features(x):
    features = []
    for i in range (len(x[0])):
        xi = [item[i] for item in x]
        features.append(xi)
    return(features)



def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):

  
    """


    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.


    dtree = {}
    if attribute_value_pairs is None:
            attribute_value_pairs = []
            x_feats = features(x)
            # print(len(x_feats))
            for i in range(0,len(x_feats)):
                unique_vals = list(set(x_feats[i]))
                # print(unique_vals)
                for val in unique_vals:
                    attribute_value_pairs.append((i,val))

    attribute_value_pairs = np.array(attribute_value_pairs)

    y_values, y_count = np.unique(y, return_counts=True)
    if len(y_values) == 1:
        return y_values[0]   

    if len(attribute_value_pairs) == 0 or depth==max_depth:
        return y_values[np.argmax(y_count)]  

    mutual_info = []
    
    for att, val in attribute_value_pairs:
        i_yx = mutual_information(np.array((x[:, att] == val).astype(int)), y)
        mutual_info.append(i_yx)

    index = mutual_info.index(max(mutual_info))
    att, val = attribute_value_pairs[index]
    attribute_value_pairs = np.delete(attribute_value_pairs, index, 0)

    # print(attribute_value_pairs)


    partitions = partition(np.array((x[:, att] == val).astype(int)))

    # print(partitions)

    for i in range(0,len(partitions)):

        x_new = np.array(x[partitions[i], :])
        y_new = np.array(y[partitions[i]])
        out = bool(i)
        dtree[(att, val, out)] = id3(x_new, y_new, attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)

    return dtree


    # raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.

    for node, successors in tree.items():
        feature = node[0]
        value = node[1]
        split = node[2]

        if(((x[feature] == value) and split) or ((x[feature] != value) and not split)):
            
                if type(successors) is dict:
                    prediction = predict_example(x, successors)
                else: 
                    prediction = successors

                return prediction

    # raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE

    return  float(1/len(y_true)) * float(np.sum(np.absolute(y_true - y_pred))) 
    
    # raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def confusion_matrix_binary(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''
  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  result = np.flip(result,0)
  result = np.flip(result,1)

  print("")
  print("\t \t   \t   Predicted Positive  \t    Predicted Negative")
  for i in range(0,len(result)):
    if(i == 0):
      print("Actual Positive \t ", end =" ")
    else:
        print("Actual Negative \t ", end =" ")
                
    for j in range(0, len(result[0])):
        print(" {} \t \t \t  ".format(int(result[i][j])), end =" " )
                
    print()
  print("")



def confusion_matrix_multiclass(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''
  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  print("")
  string = "\t \t   "
  for i in range(len(result)):
      string = string + "Pred Class {}  \t    ".format(i)

  
  print(string)
  for i in range(0,len(result)):
    print("Actual Class {}\t ".format(i), end =" ")

                
    for j in range(0, len(result[0])):
        print(" {} \t \t \t  ".format(int(result[i][j])), end =" " )
                
    print()
  print("")



if __name__ == '__main__':

    data = ['1','2','3']
    depths = [1,2,3,4,5,6,7,8,9,10]
    plt.figure(1, figsize=(16,5)).suptitle("Decision Trees")
    all_train_errors = []
    all_test_errors = []
    for t_set in data: 
        train_errors = [] 
        test_errors = []
        for dep in range(1,11):
            # Load the training data
            M = np.genfromtxt('data/monks-{}.train'.format(t_set), missing_values=0, skip_header=0, delimiter=',', dtype=int)
            ytrn = M[:, 0]
            Xtrn = M[:, 1:]


            # Load the test data
            M = np.genfromtxt('data/monks-{}.test'.format(t_set), missing_values=0, skip_header=0, delimiter=',', dtype=int)
            ytst = M[:, 0]
            Xtst = M[:, 1:]


            # Learn a decision tree of depth 3
            decision_tree = id3(Xtrn, ytrn, depth = 0, max_depth=dep)
            # print(decision_tree)
            visualize(decision_tree)

            # Compute the train error
            y_pred = [predict_example(x, decision_tree) for x in Xtrn]
            trn_err = compute_error(ytrn, y_pred)

            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)

            if(t_set=='1' and dep in [1,2]):
                cm = confusion_matrix_binary(ytst, y_pred)

            
            print('Train Error for Monks-{0} on Depth {1} = {2:4.2f}%.'.format(t_set, dep, trn_err * 100))
            train_errors.append(trn_err)
            print('Test Error for Monks-{0} on Depth {1} = {2:4.2f}%.'.format(t_set, dep, tst_err * 100))
            test_errors.append(tst_err)
        
            print("\n")
        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)


   
       
        plt.figure(int(t_set), figsize=(16,5)).suptitle("Decision Trees")
        plt.title("Monks " + t_set)
        plt.xlabel("Max Depth")
        plt.ylabel("Percetage Error")
        plt.grid()
        train_errors = all_train_errors[int(t_set)-1]
        test_errors = all_test_errors[int(t_set)-1]
        plt.plot(depths, train_errors, '*-', color='g', label='Training Error')
        plt.plot(depths, test_errors, '*-', color='r', label='Testing Error')
        plt.legend(loc="best")
        plt.savefig("Monks-{}.png".format(t_set))



    dt = tree.DecisionTreeClassifier(criterion="entropy")
    dt.fit(Xtrn, ytrn)

    y_pred_sk = dt.predict(Xtst)

    print("Confusion Matrix for the Sklearn decision tree on the Monk-1 Dataset \n")
    confusion_matrix_binary(ytst, y_pred_sk)

    dot_data = tree.export_graphviz(dt, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("Monks-1")


    data = []
    with open('data/iris.data') as f1:
        for line in f1:
            temp = line.strip('\n').split(',')
            data.append(temp)
        


    X = np.zeros((len(data), len(data[0])-1))
    y = np.zeros((len(data),1))



    for i in range(len(data[0])-1):
    
        xi = [int(round(float(val[i]))) for val in data]

        X[:, i] = xi

    X = X.astype(int)
    le = LabelEncoder()
    y = le.fit_transform([item[len(data[0])-1] for item in data])
    classes = le.classes_    

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


    # For Depth = 1

    print("Depth 1 Decision Tree on Iris Dataset")
    d1_dtree = id3(X_train, y_train, max_depth=1)
    visualize(d1_dtree)

    y_pred_train =  [predict_example(x, d1_dtree) for x in X_train]
    trn_err = compute_error(y_train, y_pred_train)

    y_pred = [predict_example(x, d1_dtree) for x in X_test]
    tst_err = compute_error(y_test, y_pred)

    confusion_matrix_multiclass( y_test, y_pred)

    print('Train Error for Iris on Depth 1 = {0:4.2f}%.'.format(trn_err * 100))

    print('Test Error for Iris on Depth 1 = {0:4.2f}%.'.format(tst_err * 100))



    # For Depth = 2

    print("Depth 2 Decision Tree on Iris Dataset")
    d2_dtree = id3(X_train, y_train, max_depth=2)
    visualize(d2_dtree)

    y_pred_train =  [predict_example(x, d2_dtree) for x in X_train]
    trn_err = compute_error(y_train, y_pred_train)

    y_pred = [predict_example(x, d2_dtree) for x in X_test]
    tst_err = compute_error(y_test, y_pred)


    confusion_matrix_multiclass( y_test, y_pred)


    print('Train Error for Iris on Depth 2 = {0:4.2f}%.'.format(trn_err * 100))

    print('Test Error for Iris on Depth 2 = {0:4.2f}%.'.format(tst_err * 100))


    # Using Sklearn
    
    print("Sklearn Decision Tree on Iris Dataset")

    
    dt = tree.DecisionTreeClassifier(criterion="entropy")
    dt.fit(X_train, y_train)

    y_pred_tst_sk = dt.predict(X_test)

    confusion_matrix_multiclass(y_test, y_pred_tst_sk)

    y_pred_train =  dt.predict(X_train)
    trn_err = compute_error(y_train, y_pred_train)
    tst_err = compute_error(y_test, y_pred_tst_sk)

    print('Train Error for Iris using Sklearn = {0:4.2f}%.'.format(trn_err * 100))

    print('Test Error for Iris using Sklearn = {0:4.2f}%.'.format(tst_err * 100))

    dot_data = tree.export_graphviz(dt, out_file=None,class_names=classes)
    graph = graphviz.Source(dot_data)
    graph.render("Iris")


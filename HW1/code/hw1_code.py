# CSC2515
# Kopal Garg, 1003063221

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from math import log2
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# question 3.a.
def load_data(data_path):
    '''
    Q 3.a
    load dataset, preprocess using a count vectorizer,
    split into train, validation and test sets
    '''
    # load in clean fake and clean real txt files
    clean_fake = pd.read_table(os.path.join(data_path, 'clean_fake.txt'), header = None)
    clean_fake['y'] = 'fake'
    clean_real = pd.read_table(os.path.join(data_path, 'clean_real.txt'), header = None)
    clean_real['y'] = 'real'
    df = clean_fake.append(clean_real)
    df_x = df[0]
    df_y = df['y']
    # preprocessing: count vectorizer 
    vectorizer = CountVectorizer()
    df_x_v = vectorizer.fit_transform(df_x)
    # split into train, validation and test sets.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    # train: 70% of the data 
    X_train, X_test, y_train, y_test = train_test_split(df_x_v, df_y, test_size= 1 - train_ratio)
    # test: 15% of initial dataset, validation: 15% of initial dataset 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    return df_x, df_y, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, df_x_v

# question 3.b.
def select_tree_model(X_train, X_val, y_train, y_val):
    '''
    Q 3.b
    train decision trees for each max_depth and criterion combination,
    compute accuracy on validation set, 
    return best model
    '''
    best_score = 0
    best_tree = None
    max_depth = [2,4,8,16,32]
    criteria = ['entropy', 'gini']
    # generate parameters 
    parameters_list = []
    for i in max_depth:
        for criterion in criteria:
            parameters_list.append({
				"max_depth": i,
				"criterion": criterion })
    for parameters in parameters_list:
        # decision Tree Classifier
        dtc = DecisionTreeClassifier(max_depth=parameters["max_depth"], 
                                                criterion=parameters["criterion"], 
                                                splitter="best",)
        # train on train data
        dtc.fit(X=X_train, y=y_train)
        # test on validation data 
        predicted = dtc.predict(X = X_val)
        correct = 0
        # compute score
        for i,j in zip(predicted, y_val):
            if i==j:
                correct+=1
        # score as a percent
        accuracy_percent = 100*(correct / len(np.asarray(y_val)))
        # print results 
        print("max_depth: {}, criterion: {}, accuracy: {:.2f}".format(
		str(parameters["max_depth"]).ljust(2),
		parameters["criterion"].ljust(7),
		accuracy_percent))
        if accuracy_percent>best_score: best_score = accuracy_percent; best_tree = dtc
    # print best results
    print("best_parameters:")
    print("best_tree: {}, best_score: {}".format(
	best_tree,
	best_score))
    return best_tree, best_score

# question 3.c. 
def best_model_accuracy(X_test, y_test, best_tree):
    '''
    Q 3.c
    compute test accuracy of best tree
    '''
    predicted = best_tree.predict(X=X_test)
    correct = 0
    # compute score
    for i,j in zip(predicted, y_test):
        if i==j:
            correct+=1
    # score as a percent
    accuracy_percent = 100*(correct / len(np.asarray(y_test)))
    # print results 
    print("test accuracy: {:.2f}".format(accuracy_percent))
    return accuracy_percent

def visualize_tree(best_tree, vectorizer):
    '''
    Q 3.c
    plot the first two layers of the best tree
    '''
    fig, axes = plt.subplots(figsize=(12,12))
    tree.plot_tree(best_tree, 
    max_depth=2, 
    feature_names = vectorizer.get_feature_names(), 
    class_names = best_tree.classes_,
    filled=True,
    fontsize=15)
    fig.savefig('images/Q3_c_tree.png')

def log2_(x):
    if x ==0:
        return(0)
    else:
        return(log2(x))
# question 3.d.
def compute_information_gain(vectorizer, df_y, df_x_v, feature_name):
    '''
    Q 3.d
    compute the information gain for given feature 
    '''
    def entropy(y):
        real =  y.where(y == 'real').dropna()
        pr = len(real)/len(y)
        H = -(pr*log2_(pr))-((1-pr)*log2_(1-pr))
        return H

    features = vectorizer.get_feature_names()

    if feature_name in features: 
        id = features.index(feature_name)
    else:
        print("Information Gain for {} is {:.3f}".format(feature_name, 0))
        return None
    df_x_v_array = df_x_v.toarray()
   
    # parent node entropy
    H_parent = entropy(df_y)

    # right split entropy
    right_y = df_y[df_x_v_array[:,id] >= 0.5]
    H_right = entropy(right_y)

    # left split entropy
    left_y = df_y[df_x_v_array[:,id] < 0.5]
    H_left = entropy(left_y)

    # information gain
    IG = H_parent - (((len(left_y)/len(df_y))*H_left) + ((len(right_y)/len(df_y))*H_right))

    print("Information Gain for {} is {:.3f}".format(feature_name, IG))

    return IG

# question 3.e.
def select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test, data_path):
    '''
    Q 3.e
    classify between real and fake news using a KNN classifier 
    '''
    # load in clean fake and clean real txt files
    clean_fake = pd.read_table(os.path.join(data_path, 'clean_fake.txt'), header = None)
    clean_fake['y'] = 'fake'
    clean_real = pd.read_table(os.path.join(data_path, 'clean_real.txt'), header = None)
    clean_real['y'] = 'real'
    df = clean_fake.append(clean_real)
    df_x = df[0]
    df_y = df['y']
    # preprocessing: count vectorizer 
    vectorizer = CountVectorizer()
    df_x_v = vectorizer.fit_transform(df_x)
    # split into train, validation and test sets.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    # train: 70% of the data 
    X_train, X_test, y_train, y_test = train_test_split(df_x_v, df_y, test_size= 1 - train_ratio)
    # test: 15% of initial dataset, validation: 15% of initial dataset 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    # KNN
    train_error = []
    val_error = []
    best_val_acc = -1
    best_model = None
    for k in range(1, 21):
        knc = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        pred_train = knc.predict(X_train)
        pred_val = knc.predict(X_val)
        # training and validation accuracy
        train_acc_k = metrics.accuracy_score(y_train, pred_train)
        val_acc_k = metrics.accuracy_score(y_val, pred_val)
        if val_acc_k > best_val_acc: best_val_acc = val_acc_k; best_model = knc
        print("Train Accuracy:", metrics.accuracy_score(pred_train, y_train))
        print("Validation Accuracy:", metrics.accuracy_score(pred_val, y_val))
        # training and validation error
        error_train = 0
        error_val = 0
        for i,j in zip(pred_train, y_train):
            if i!=j:
                error_train+=1
        for i,j in zip(pred_val, y_val):
            if i!=j:
                error_val+=1
        train_error.append(error_train)
        val_error.append(error_val)
    
    print("Best validation accuracy {}:".format(best_val_acc))
    # compute test accuracy using best KNN model
    print("Test accuracy: {}".format(metrics.accuracy_score(y_test, best_model.predict(X_test))))
    # plot the validation and train accuracy curves 
    fig, axes = plt.subplots(figsize=(12,12))
    axes.plot(range(1,21), train_error, label = 'train', color ='cyan', marker='o')
    axes.plot(range(1,21), val_error, label = 'validation', color = 'orange', marker='o')
    axes.legend()
    axes.invert_xaxis()
    axes.set_xlabel("k (number of nearest neighbors)")
    axes.set_ylabel("error")
    fig.savefig('images/Q3_e_KNN.png')

    
def main():
    data_path = '/Users/kopalgarg/Documents/GitHub/CSC2515/HW1/data/'
    df_x, df_y, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer,df_x_v = load_data(data_path)
    best_tree, best_score = select_tree_model(X_train, X_val, y_train, y_val)
    best_model_accuracy(X_test, y_test, best_tree)
    visualize_tree(best_tree, vectorizer)
    # Top most split keyword
    compute_information_gain(vectorizer, df_y, df_x_v, 'donald')
    # Other keywords
    compute_information_gain(vectorizer, df_y, df_x_v, 'hillary')
    compute_information_gain(vectorizer, df_y, df_x_v, 'the')
    compute_information_gain(vectorizer, df_y, df_x_v, 'trump')
    compute_information_gain(vectorizer, df_y, df_x_v, 'dogs')
    # KNN
    select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test, data_path)

if __name__ == "__main__":
	main()
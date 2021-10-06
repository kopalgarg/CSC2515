import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pprint import pprint
from sklearn import tree

# question 3.a.
def load_data(data_path):
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
    best_score = 0
    best_tree = None
    max_depth = [2,4,6,8,10,12]
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
        dtc = DecisionTreeClassifier(max_depth=parameters["max_depth"], criterion=parameters["criterion"], splitter="random",)
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

def plot_tree(best_tree, vectorizer):
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(best_tree, 
    max_depth=2, 
    feature_names = vectorizer.get_feature_names(), 
    class_names = best_tree.classes_)
    fig.savefig('images/Q3_c_tree.png')

# question 3.d.
def compute_information_gain(X_train, y_train, vectorizer, df_x, df_x_v, feature_name):
    features = vectorizer.get_feature_names()
    if feature_name in features: id = features.index(feature_name)
    df_x_v_array = df_x_v.toarray()
    print(df_x_v_array[:,id])

    return 0

def main():
    df_x, df_y, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer,df_x_v = load_data('/Users/kopalgarg/Documents/GitHub/CSC2515/HW1/data/')
    best_tree, best_score = select_tree_model(X_train, X_val, y_train, y_val)
    best_model_accuracy(X_test, y_test, best_tree)
    plot_tree(best_tree, vectorizer)
    compute_information_gain(X_train, y_train, vectorizer, df_x, df_x_v, 'trump')

if __name__ == "__main__":
	main()
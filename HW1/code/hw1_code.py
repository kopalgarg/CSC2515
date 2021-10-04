import numpy as np
import pandas as pd
from matplotlib import pyplot
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pprint import pprint

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

    return X_train, X_val, X_test, y_train, y_val, y_test

# question 3.b.
def select_tree_model(X_train, X_val, y_train, y_val):
    max_depth = [2,4,6,8,10,12]
    criteria = ['entropy', 'gini']
    parameters = dict(dec_tree__criterion=criteria,
                      dec_tree__max_depth=max_depth)
    # decision tree
    dt = DecisionTreeClassifier()
    pipe = Pipeline(steps=[('dec_tree', dt)])
    # grid search
    clf_gs = GridSearchCV(pipe, parameters)
    clf_gs.fit(X_train, y_train)
    # best model parameters 
    criterion_best = clf_gs.best_estimator_.get_params()['dec_tree__criterion']
    max_depth_best = clf_gs.best_estimator_.get_params()['dec_tree__max_depth']
    # best model
    clf = clf_gs.best_estimator_.get_params()['dec_tree']
    # predictions 
    predictions = clf.predict(X_val)
    # accuracy score computation
    accuracy_sc = accuracy_score(y_val, predictions)

    means = clf_gs.cv_results_['mean_test_score']
    stds = clf_gs.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))

    return clf

def select_model(x_train, x_val,y_train, y_val):     
    #fit the training datasets to Classifier 
    acc_rate = 0
    depth_test = pd.Series([5,10,20,30,40], index = [0,1,2,3,4])
    cri_test = pd.Series(['gini','entropy'], index = [0,1])
    for a in range (cri_test.count()):
        for b in range (depth_test.count()):
            clf = DecisionTreeClassifier(max_depth=depth_test[b],criterion=cri_test[a])
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_val)
            real = np.array(y_val)
            count = 0
            
            #calculate accurancy of validation data for each combination
            for i in range (len(real)):
                if predictions[i] == real[i]:
                    count =count +1
                    acc_rate1 = count/len(real)
            print('Accurancy is', acc_rate1, 'with decision tree depth equals to', depth_test[b], 'and criterion is', cri_test[a])
            
            # find the classifier with best performance (biggest accurancy) 
            if acc_rate1 >= acc_rate:
                acc_rate = acc_rate1
                depth = depth_test[b]
                cri = cri_test[a]
                classifier = clf
    print('The best classifier has accurancy of', acc_rate, 'with decision tree depth equals to', depth, 'and criterion is', cri)

X_train, X_val, X_test, y_train, y_val, y_test = load_data('/Users/kopalgarg/Documents/GitHub/CSC2515/HW1/data/')
select_tree_model(X_train, X_val, y_train, y_val)
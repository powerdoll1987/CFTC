# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:23:29 2016

@author: wei.liu
"""
from numpy import *
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from gplearn.genetic import SymbolicTransformer
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble.partial_dependence import plot_partial_dependence
#from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sknn.mlp import Classifier, Layer





#treePlotter.createPlot(myTree)

class RandomForestLearner():
    def __init__(self, k=50):
        self.k = k
        self.forest = []

    def addEvidence(self, train,test):
        self.xdata = train
        self.ydata = test
        self.data = column_stack((self.xdata, self.ydata))
        length = math.ceil(self.xdata.shape[0]*.8)
        indeces = arange(self.xdata.shape[0])
        for i in range(0,self.k):
            random.shuffle(indeces)
            tree = regTrees.createTree(self.xdata[indeces[0:length],:],regTrees.modelLeaf,regTrees.modelErr,(1,20))
            self.forest.append(tree) 
    
    def forcasting(self):   
        somme = 0        
        for i in range(0,self.k):
            yHat = regTrees.createForeCast(self.forest[i],self.ydata[:,:-1],regTrees.modelTreeEval)
            somme = somme + yHat
        return (somme / self.k)


if __name__ == '__main__':   

    if 1:
        print('1.Data Preprocessing')
#        df_chg = pd.read_excel('posCHG_M.xls')
#        df_pct = pd.read_excel('posCHG_M.xls')
#        df_pct_zs = pd.read_excel('posCHG_M.xls')
#        data = pd.concat([df_chg,df_pct,df_pct_zs],axis = 0,join = 'inner')
#        data.to_excel('cftc.xlsx')
        #data = pd.read_excel('cftc.xlsx')
        data = pd.read_excel('ZS_CHG.xls')
        data = data.dropna()
        ind = [ True,  True, False, False,  True,  True, False,  True,  True,
       False, False, False, False, False, False, False, False, True, True]
        if 0:
            data = data[data.pivots != 0]
            trainMat = data.as_matrix()[:150,:]
            testMat = data.as_matrix()[:,:]    
                
        if 1:
            
            trainMat = data.as_matrix()[:700,:]
            testMat = data.as_matrix()[700:,:]
        
        X = data.as_matrix()[:,:-2]
        Y = data.as_matrix()[:,-2]
    if 1:
        nn = Classifier(
        layers=[
            Layer("Rectifier", units=200),
            Layer("Softmax")],
            learning_rate=0.02,
            n_iter=10)
        nn.fit(trainMat[:,:-2], trainMat[:,-2])
        res = nn.predict(testMat[:,:-2])

        
    
    
    
    
    if 0:
        print('Logistic')
        clf = LogisticRegression( solver= 'liblinear' , penalty = 'l1' ,n_jobs = 4)  
        clf.fit(trainMat[:,:-2], trainMat[:,-2])
        g = clf.predict(testMat[:,:-2])
        print(clf.score(testMat[:,:-2],testMat[:,-2]))
        prob = clf.predict_proba(data.as_matrix()[:,:-2])
        
        print(corrcoef(g,testMat[:,-2],rowvar=0)[0,1])
        
        
    
    
    
    if 0:
        print('2.Feature Extraction')
        
        gp = SymbolicTransformer(generations=100, population_size=200,
                                 hall_of_fame=2, n_components=2,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=1,
                                 random_state=0, n_jobs=3)
        gp.fit(trainMat[:,:-1], trainMat[:,-1])
    
    if 0:
        print('SVC')
        clf = svm.SVC(decision_function_shape='ovo',probability = True,kernel='rbf')
        clf.fit(trainMat[:,:-2], trainMat[:,-2])
        print(clf.score(testMat[:,:-2],testMat[:,-2]))
        res = clf.predict(testMat[:,:-2])
        print(corrcoef(res,testMat[:,-2],rowvar=0)[0,1])
        prob = clf.predict_proba(data.as_matrix()[:,:-2])
    
    
    if 0:
        #clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf = tree.DecisionTreeClassifier()
        scores = cross_val_score(clf, X, Y)
        print(scores.mean())   
        clf.fit(trainMat[:,:-2], trainMat[:,-2])
        g = clf.predict(testMat[:,:-2])
        print(corrcoef(g,testMat[:,-2],rowvar=0)[0,1])
        print("MSE: %.4f" % mean_squared_error(testMat[:,-2], g))
        
        
        
    if  0:        
        svr = svm.SVR(kernel='rbf', C=1e3)
        svr.fit(trainMat[:,:-2], trainMat[:,-2])
        print(svr.score(testMat[:,:-2],testMat[:,-2]))
        
        
        
        
        
        
        
    if 0:
        clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
        scores = cross_val_score(clf, X, Y)
        print(scores.mean())   
        clf.fit(trainMat[:,:-2], trainMat[:,-2])
        g = clf.predict(testMat[:,:-2])
        print(corrcoef(g,testMat[:,-2],rowvar=0)[0,1])
        print("MSE: %.4f" % mean_squared_error(testMat[:,-2], g))
        
        
    if 0:
        print('3.GBRT')
        params = {'n_estimators': 500, 'max_depth': 8, 'min_samples_split': 1,
                  'learning_rate': 0.01, 'loss': 'ls','verbose':1}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(trainMat[:,:-2], trainMat[:,-2])
        g = clf.predict(testMat[:,:-2])
        print(corrcoef(g,testMat[:,-2],rowvar=0)[0,1])
        print("MSE: %.4f" % mean_squared_error(testMat[:,-2], g))
        
        
        features = [0, 19, 1, 2, (19, 1)]
        fig, axs = plot_partial_dependence(clf, data.as_matrix()[:,:-2],features_jobs=3, grid_resolution=50)  
    
    if 0:    
        print('4.GBRT + GP')
            
        gp_features = gp.transform(trainMat[:,:-1])
        data_train = hstack((trainMat[:,:-1], gp_features))
        
        gp_features = gp.transform(testMat[:,:-1])
        data_test = hstack((testMat[:,:-1], gp_features))
        
        params = {'n_estimators': 500, 'max_depth': 8, 'min_samples_split': 1,
                  'learning_rate': 0.05, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(data_train, trainMat[:,-1])
        g = clf.predict(data_test)
        print(corrcoef(g,testMat[:,-1],rowvar=0)[0,1])
        print("MSE: %.4f" % mean_squared_error(testMat[:,-1], g))



import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def data_segmentation(data,segnum):
    np.random.seed(42)
    total_num,_=data.shape
    data=np.array(data.iloc[:,:])
    shuffled_indices = np.random.permutation(total_num)
    data=data[shuffled_indices,:]
    single_num=int(total_num/segnum)
    out=[]
    for i in range(segnum-1):
        out.append(data[i*single_num:(i+1)*single_num,:])
    out.append(data[(segnum-1)*single_num:(total_num)*single_num,:])
    return out

def segmentation2data(data,segnum,testindex):
    _,y=data[0].shape
    traindata=np.zeros((0,y))
    for i in range(segnum):
        if i==testindex:
            testdata = data[i]
        if i!=testindex:
            traindata = np.concatenate((traindata,data[i]),axis=0)
    return traindata,testdata

def bootstrap(prob,pred,label, B, index):
    prob_array = np.array(prob)
    pred_array = np.array(pred)
    n = len(prob_array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        prob_sample = prob_array[index_arr]
        pred_sample = pred_array[index_arr]
        label_sample = label[index_arr]
        fpr,tpr, _ = roc_curve(label_sample,prob_sample)
        confusion_sample = confusion_matrix(label_sample,pred_sample)
        TP_sample = confusion_sample[1, 1]
        TN_sample = confusion_sample[0, 0]
        FP_sample = confusion_sample[0, 1]
        FN_sample = confusion_sample[1, 0]

        if index == 'auc':
            sample_result = metrics.auc(fpr,tpr)
        if index == 'acc':
            sample_result = (TP_sample+TN_sample)/(TP_sample+TN_sample+FP_sample+FN_sample)
        if index == 'sen':
            sample_result = TP_sample / float(TP_sample+FN_sample)
        if index == 'spe':
            sample_result = TN_sample / float(TN_sample+FP_sample)
        
        sample_result_arr.append(sample_result)
        
    sample_result_arr=np.array(sample_result_arr)
    
    mean=np.mean(sample_result_arr)
    std=np.std(sample_result_arr)

    lower = mean-1.96*std
    higher = mean+1.96*std
    return lower, higher

def model_classification(data_train,data_test):  
    x_train = data_train.iloc[:,1:]
    x_test = data_test.iloc[:,1:]
    y_train = data_train.iloc[:,0]
    y_test = data_test.iloc[:,0]
    
    #Lasso
    alphas = np.logspace(-2,3,50)
    model_lassoCV = LassoCV(alphas = alphas,cv = 10,max_iter = 100000,random_state = 42).fit(x_train,y_train)
    coef = pd.Series(model_lassoCV.coef_, index = x_train.columns)
    index_lasso = coef[coef != 0].index
    x_train_lasso = x_train[index_lasso]
    
    #RFE
    model = RFE(LogisticRegression(solver='liblinear',random_state = 42,class_weight = balanced),step=1)
    param_grid = {'estimator__penalty': ['l1','l2'],'estimator__C': [1e-2, 1e-1,0.5, 1, 5,10],'n_features_to_select' :[10,13,15,16,18,20,22,24,26]}    
    grid_search = GridSearchCV(model, param_grid,cv=10)
    grid_search.fit(x_train_lasso, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()
    classifier = LogisticRegression(penalty = best_parameters['estimator__penalty'],C = best_parameters['estimator__C'],solver='liblinear',random_state = 42)
    selector = RFE(classifier, n_features_to_select = best_parameters['n_features_to_select'], step=1).fit(x_train_lasso, y_train)
        
    index_rfe = np.where(selector.ranking_ == 1)[0]
    x_train_rfe = x_train.iloc[:,index_rfe]
    x_test_rfe = x_test.iloc[:,index_rfe]
    classifier_model = classifier.fit(x_train_rfe, y_train)
    prob = classifier_model.predict_proba(x_test_rfe)
    y_pred = classifier_model.predict(x_test_rfe)

    return prob,y_pred,y_test,y_train,x_train_rfe,x_test_rfe

B = int(1e6)

data = pd.read_excel(r"C:\...\data.xlsx")
x = data.iloc[:,1:]
label = data.iloc[:,0]
colname = x.columns
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x.columns = colname
data = pd.concat([label,x],axis = 1)

prob_all = []
y_pred_all = []
y_test_all = pd.DataFrame()
for data_list in range(5):
    segmentationdata=data_segmentation(data,5)
    train_data,test_data=segmentation2data(segmentationdata,5,data_list)
    colname = data.columns
    train_data = pd.DataFrame(train_data,columns = colname)
    test_data = pd.DataFrame(test_data,columns = colname)
    
    prob,y_pred,y_test,y_train,x_train_rfe,x_test_rfe = model_classification(train_data,test_data)

    prob_all = np.append(prob_all,prob)
    y_pred_all = np.append(y_pred_all,y_pred)
    y_test_all = pd.concat([y_test_all,y_test],axis = 0)
    
    
prob_all = prob_all.reshape(-1,2)
y_pred_all = y_pred_all.reshape(-1,1)

fpr, tpr, threshold = roc_curve(y_test_all, prob_all[:,1], pos_label = 1)
auc_score = roc_auc_score(y_test_all, prob_all[:,1])
confusion = confusion_matrix(y_test_all,y_pred_all)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
acc = (TP+TN)/(TP+TN+FP+FN)
sen = TP / float(TP+FN)
spe = TN / float(TN+FP)

prob_all = pd.DataFrame(prob_all)
y_pred_all = pd.DataFrame(y_pred_all,columns=['y_pred'])

y_test_all =y_test_all.reset_index(drop=True)
result = pd.concat([prob_all,y_test_all,y_pred_all],axis = 1)

index_list = ['auc','acc','sen','spe']
for index in index_list:
    if index == 'auc':
        print('auc:',auc_score)
    if index == 'acc':
        print('acc:',acc)
    if index == 'sen':
        print('sen:',sen)
    if index == 'spe':
        print('spe:',spe)
    print('CI:',bootstrap(result.iloc[:,1],result.iloc[:,3],result.iloc[:,2],B,index))
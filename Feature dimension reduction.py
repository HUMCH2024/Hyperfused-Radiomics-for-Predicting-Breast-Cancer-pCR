import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr

#Import extracted radiomics features
file_path = r'C:\...\feature.xlsx'
df = pd.read_excel(file_path)
data_feature = df.iloc[:,1:]
label = df.iloc[:,0]

#Variance
# Calculate the variance of each column feature
variances = data_feature.var()

# Sort the variance results in descending order
sorted_variances = variances.sort_values(ascending=False)

# Calculate the number of features to be deleted (last 25%)
num_features_to_drop = int(len(sorted_variances) * 0.25)

# Retrieve the index of features to be retained
features_to_keep = sorted_variances.index[:-num_features_to_drop]

# Extract columns from DataFrame based on index
filtered_df = df[features_to_keep]

#Pearson
data_feature_std = StandardScaler().fit_transform(filtered_df)
data_feature_std = pd.DataFrame(data_feature_std)
data_feature_std.columns = data_feature.columns
pearson = data_feature_std.corr()
colname = pearson.columns
pearson1=np.triu(pearson,k=1)
pearson1=pd.DataFrame(pearson1)
pearson1.columns = pearson.columns
pearson1.index = pearson.index
index_list1= []
index_list2=[]
for i in range(pearson1.shape[1]):
    index_list1= index_list1+(list(pearson1.loc[(pearson1[colname[i]]>0.75) & (pearson1[colname[i]]<=1)].index))
    index_list2= index_list1+(list(pearson1.loc[(pearson1[colname[i]]< -0.75) & (pearson1[colname[i]]>= -1)].index))
index_list = index_list1 + index_list2
res = []
for n in index_list:
    if i not in res:
        res.append(n)
data_pearson = data_feature.drop(res,axis = 1)

#t/u test 
data1 = data_pearson.iloc[:,1:]
data1_name = data1.columns
data1 = StandardScaler().fit_transform(data1)
data1 =  pd.DataFrame(data1)
data1.columns = data1_name
new = pd.concat([data_pearson.iloc[:,0],data1],axis=1) 
data_pcr = new[new['label'] == 1].iloc[:,2:]
print(data_pcr.shape)
data_npcr = new[new['label'] == 0].iloc[:,2:]
print(data_npcr.shape)
colnames = data_pcr.columns
drop_index = []
for colname in colnames:
    pcr_nomal_p_value = stats.shapiro(data_pcr[colname])[1]
    npcr_nomal_p_value = stats.shapiro(data_npcr[colname])[1]
    std_p_value = stats.levene(data_pcr[colname],data_npcr[colname])[1]
    if pcr_nomal_p_value > 0.05 and npcr_nomal_p_value > 0.05:
        if std_p_value > 0.05:
            t_pvalue = stats.ttest_ind(data_npcr[colname],data_pcr[colname])[1]

        else:
            t_pvalue = stats.ttest_ind(data_npcr[colname],data_pcr[colname],equal_var=False)[1]
    else:
        t_pvalue = stats.mannwhitneyu(data_npcr[colname],data_pcr[colname],alternative='two-sided')[1]
    if t_pvalue > 0.05 or t_pvalue == 'nan':
        drop_index.append(colname)
    if t_pvalue < 0.05:
        print("t_or_u_pvalue:",t_pvalue)

print(drop_index)
data_tutest = data_pearson.drop(drop_index,axis=1)
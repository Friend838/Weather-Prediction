#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/user/Desktop/data_science/train.csv')
print('Size of weather data frame is :',df.shape)
df[0:5]


# In[2]:


df.count().sort_values()


# In[3]:


df = df.drop(columns = ['Attribute7','Attribute6','Attribute18', 'Attribute19','Attribute2','Attribute1'], axis=1)
df.shape


# In[4]:


df = df.dropna(how='any')
df.shape


# In[5]:


#讓差異大的數據去除掉

from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df = df[(z < 3).all(axis=1)]
df[0:5]


# In[6]:


df['Attribute22'].replace({'No': 0, 'Yes': 1},inplace = True)
df['Attribute23'].replace({'No': 0, 'Yes': 1},inplace = True)

categorical_columns = ['Attribute8', 'Attribute10', 'Attribute11']
for col in categorical_columns:
    print(np.unique(df[col]))

df = pd.get_dummies(df, columns=categorical_columns)


# In[7]:


df[0:5]


# In[8]:


X = df[['Attribute3','Attribute4','Attribute5','Attribute9','Attribute12','Attribute13','Attribute14','Attribute15','Attribute16','Attribute17','Attribute20','Attribute21','Attribute22']]
y = df[['Attribute23']]


# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from imblearn.over_sampling import SMOTE

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

clf_logreg = LogisticRegression(random_state=0)
clf_logreg.fit(X_train_res,y_train_res)
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
clf_rf.fit(X_train_res,y_train_res)
y_pred = clf_rf.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train_res,y_train_res)
y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[16]:


from sklearn import svm
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

clf_svc = svm.SVC(kernel='linear')
clf_svc.fit(X_train_res,y_train_res)
y_pred = clf_svc.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)


# In[17]:


df2 = pd.read_csv('C:/Users/user/Desktop/data_science/test.csv')
df2 = df2[['Attribute3','Attribute4','Attribute5','Attribute9','Attribute12','Attribute13','Attribute14','Attribute15','Attribute16','Attribute17','Attribute20','Attribute21','Attribute22']]
df2


# In[18]:


df2['Attribute22'].replace({'No': 0, 'Yes': 1},inplace = True)
df2


# In[19]:


df2_pred = clf_logreg.predict(df2)
df2_pred


# In[20]:


df2_pred = pd.DataFrame(df2_pred, dtype = np.int)
df2_pred


# In[21]:


output = pd.DataFrame(index = df2.index, columns = ['id', 'ans'], )
output['id'] = df2.index + 0.0
output['ans'] = df2_pred
output


# In[22]:


filename = 'output_test.csv'

output.to_csv(filename,index=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


pd.pandas.set_option('display.max_columns',None)


# In[6]:


data = pd.read_csv("C:/Users/kalpana/Downloads/train.csv")


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.info() 


# In[10]:


data.isnull().sum() 


# In[11]:


data.duplicated().any()


# # Visualization on data
# Discover how many mobiles Have bluetooth or not

# In[12]:


plt.figure(figsize=(14,6))
sns.countplot(data = data ,x= data['blue'])


# it seems like mobiles not have Bluetooth are more than mobiles has Bluetooth

# # Discover the Front Camera mega pixels

# In[13]:


data.fc.value_counts().head()


# In[14]:


plt.figure(figsize=(14,6))
data.fc.value_counts().head().plot(kind = 'bar')
plt.show()


# 

# In[15]:


data.corr().T


# Number of cores of processor

# In[16]:


data['n_cores'].value_counts()


# In[17]:


plt.figure(figsize=(14,6))
sns.barplot(x = "n_cores",  y = "price_range", data = data)


# Mobiles Has 4G or not

# In[18]:


data.four_g.value_counts()


# In[19]:


plt.figure(figsize=(14,6))
data.four_g.value_counts().plot(kind = 'bar')


# Battery Power

# In[20]:


plt.figure(figsize=(14,6))
sns.boxplot(data.battery_power)


# In[21]:


data['price_range'].value_counts()


# In[22]:


plt.figure(figsize=(14,6))
sns.barplot(data = data , x  =data['price_range'] , y =data.price_range.index )


# Time talk and prica range

# In[23]:


plt.figure(figsize=(14,6))
sns.barplot(data = data , x ='talk_time' , y= 'price_range' )


# Price Range vs all numerical factor

# In[24]:


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Price Range vs all numerical factor')
sns.countplot(ax=axes[0, 0], data=data, x='three_g',palette='RdPu')
sns.countplot(ax=axes[0, 1], data=data, x='touch_screen',palette='RdPu')
sns.countplot(ax=axes[0, 2], data=data, x='four_g',palette='RdPu')
sns.countplot(ax=axes[1, 0], data=data, x='wifi',palette='RdPu')
sns.countplot(ax=axes[1,1],data = data, x ='fc' ,palette='RdPu')
sns.countplot(ax=axes[1,2],data = data, x ='dual_sim',palette='RdPu' )
plt.show()


# In[25]:


x = data.drop('price_range',axis=1)
y = data['price_range']


# # Feature Selection
# Apply SelectKBest Algorithm

# In[26]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[27]:


ordered_rank_features = SelectKBest(score_func=chi2,k=20)
ordered_feature = ordered_rank_features.fit(x,y)


# In[28]:


dfscores = pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns = pd.DataFrame(x.columns)


# In[29]:


features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']
features_rank


# Take top 10 features variables.

# In[30]:


features_rank.nlargest(10,'Score')


# # Feature Importance
# This technique gives you a score for each feature of your data, the higher the score more relevant it is

# In[31]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(x,y)


# In[32]:


print(model.feature_importances_)


# In[33]:


ranked_features=pd.Series(model.feature_importances_,index=x.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()


# Information Gain

# In[34]:


from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(x,y)
mutual_data=pd.Series(mutual_info,index=x.columns)
mutual_data.sort_values(ascending=False)


# In[35]:


data = data.drop(['wifi','touch_screen','three_g','talk_time','sc_w','dual_sim','four_g','int_memory','blue','n_cores','mobile_wt','m_dep','fc'],axis=1)


# In[36]:


data.head()


# In[37]:


x = data.drop('price_range',axis=1)
y = data['price_range']


# In[38]:


print(x.shape)
print(y.shape)


# Split the dataset into train and test

# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[40]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Feature Scaling

# In[41]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[42]:


X_train


# # Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[44]:


y_pred = classifier.predict(X_test)


# In[45]:


print(y_pred)


# In[46]:


from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc1}")


# # SVM

# In[47]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# # Predict the tset set result

# In[48]:


y_pred = classifier.predict(X_test)


# In[49]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc2 = accuracy_score(y_test, y_pred)


# In[50]:


print(f"Accuracy score: {acc2}")


# # Training the K-NN model on the Training set

# In[51]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[69]:


y_pred = classifier.predict(X_test)


# In[72]:


from sklearn.metrics import  accuracy_score
acc3 = accuracy_score(y_test, y_pred)


# In[73]:


print(f"Accuracy score: {acc3}")


# # Training the Naive Bayes on the Training set

# In[74]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[75]:


y_pred = classifier.predict(X_test)


# In[76]:


from sklearn.metrics import  accuracy_score
acc4 = accuracy_score(y_test, y_pred)


# In[77]:


print(f"Accuracy score : {acc4}")


# # Training Decision Tree Classification on Train set

# In[78]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[79]:


y_pred = classifier.predict(X_test)


# In[80]:


from sklearn.metrics import  accuracy_score
acc5 = accuracy_score(y_test, y_pred)


# In[81]:


print(f"Accuracy score: {acc5}")


# # Create visualization for all model with their Accuracy

# In[83]:


mylist=[]
mylist2=[]
mylist.append(acc1)
mylist2.append("Logistic Regression")
mylist.append(acc2)
mylist2.append("SVM")
mylist.append(acc3)
mylist2.append("KNN")
mylist.append(acc4)
mylist2.append("Naive Bayes")
mylist.append(acc5)
mylist2.append("DTR")


# In[84]:


plt.rcParams['figure.figsize']=8,6
sns.set_style("darkgrid")
ax = sns.barplot(x=mylist2, y=mylist, palette = "rocket", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[ ]:





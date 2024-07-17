#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[26]:


df = pd.read_csv('C:\\Users\\ASUS\\Downloads\\disney_plus_titles.csv')
df.head()


# In[27]:


df.info()


# In[28]:


df.isnull().sum()


# In[29]:


df.describe()


# In[30]:


if 'title' in df.columns and 'description' in df.columns:
    df['description_sentiment']=df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.hist(df['description_sentiment'],bins=20,edgecolor='b')
    plt.title('Sentiment Polarity Distribution of Description')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.show()
    print(df['description_sentiment'].describe())
else:
    print("Columns 'title'or 'description' not found in the dataset.")


# In[31]:


if 'title' in df.columns and 'rating' in df.columns:
    le_type=LabelEncoder()
    le_rating=LabelEncoder()
    df['type_encoded']=le_type.fit_transform(df['type'])
    df['rating_encoded']=le_rating.fit_transform(df['rating'].astype(str))
    X=df[['type_encoded','rating_encoded']].dropna()
    kmeans=KMeans(n_clusters=5,random_state=42)
    df['cluster']=kmeans.fit_predict(X)
    pca=PCA(n_components=2)
    X_pca=pca.fit_transform(X)
    plt.scatter(X_pca[:,0],X_pca[:,1],c=df['cluster'],cmap='viridis')
    plt.title('Clustering of titles based on Type ans Rating')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='cluster')
    plt.show()
    print("cluster centers:\n",kmeans.cluster_centers_)
    cluster_counts=df['cluster'].value_counts().sort_index()
    print("\nCount of titles in Each Cluster:")
    print(cluster_counts)
    print("\nTop Types and Ratings in Each Cluster:")
    for cluster in range(5):
        cluster_data=df[df['cluster']==cluster]
        top_types=cluster_data['type'].value_counts().head(3)
        top_ratings=cluster_data['rating'].value_counts().head(3)
        print(f"\nCluster {cluster}:")
        print("Top_Types:")
        print(top_types)
        print("Top_Ratings")
        print(top_ratings)
else:
    print("Columns 'type' or 'rating' not found in the dataset.")


# In[32]:


if 'date_added' in df.columns:
    df['date_added']=pd.to_datetime(df['date_added'],errors='coerce')
    df['year_added']=df['date_added'].dt.year
    df['month_added']=df['date_added'].dt.month
    df.groupby('year_added').size().plot(kind='bar')
    plt.title('Number of Titles Added Per Year')
    plt.xlabel('Year_Added')
    plt.ylabel('Number of Titles')
    plt.show()
    df.groupby("month_added").size().plot(kind='bar')
    plt.title("Number of titles added per month")
    plt.xlabel('Month_Added')
    plt.ylabel("Number of titles")
    plt.show()
else:
    print("column 'date_added' not found in dataset.")


# In[33]:


df.to_csv('C:\\Users\\ASUS\\Downloads\\disney_plus_titles.csv')
print("processed data saved to 'disney_plus_titles_processed_csv'")


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')


# In[36]:


df = pd.read_csv('C:\\Users\\ASUS\\Downloads\\disney_plus_titles.csv')

df['date_added'] = pd.to_datetime(df [ 'date_added'], errors='coerce')

df = df.dropna(subset=['date_added'])

df.set_index ('date_added', inplace=True)

monthly_titles =df.resample ('M'). size ()
monthly_titles.head ()


# In[37]:


decomposition=seasonal_decompose(monthly_titles,model='additive')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(12,10),sharex=True)
decomposition.observed.plot(ax=ax1)
ax1.set_ylabel('observed')

decomposition.trend.plot(ax=ax2)
ax2.set_ylabel('trend')

decomposition.seasonal.plot(ax=ax3)
ax3.set_ylabel('seasonal')

decomposition.resid.plot(ax=ax4)
ax4.set_ylabel('residual')

plt.xlabel('Date')
plt.tight_layout ()
plt.show ()


# In[38]:


model = ExponentialSmoothing ( monthly_titles, seasonal='add' , seasonal_periods=12)
fit=model.fit()

forecast =fit.forecast(12)

plt.figure(figsize=(12,5))
plt.plot(monthly_titles, label='observed')
plt.plot(forecast, label ='Forecast', linestyle ='--')
plt.xlabel ('Date')
plt.ylabel ('Number of titles added ')
plt.title('Monthly titles added to Disney plus ')
plt.legend()
plt.show()


# In[ ]:





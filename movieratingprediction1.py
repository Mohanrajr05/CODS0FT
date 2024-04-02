# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')
df.head()

# %%
def dataoverview (df,message):
    print(f'{message}:\n')
    print("Rows:", df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nFeatures:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())

# %%
dataoverview(df, 'Overview of the training dataset')

# %%
df.isna().sum()

# %%
df.info()

# %%
#genre, director, and actors values counts
df['Genre'].value_counts()


# %%
df['Director'].value_counts()

# %%

df.head(10)

# %%
# As we are going to predict movie ratings based on fdf.dropna(subset=['Name','Year','Duration','Votes','Rating'],inplace=True)eatures, we need to remove null values from features that can directly influence the results.
df.dropna(subset=['Name','Year','Duration','Votes','Rating'],inplace=True)
df.isna().sum()

# %%
# Remove parentheses from 'Year' column and convert to integer
df['Year'] = df['Year'].str.strip('()').astype(int)

# %%
# Remove commas from 'Votes' column and convert to integer
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

# %%
# Remove min from 'Duration' column andDurationonvert to integer
df['Duration'] = df['Duration'].str.replace('min', '').astype(int)

# %%

df.info()

# %%

df.describe()

# %%
# Drop Genre column 
df.drop('Genre',axis=1,inplace=True)

# %%
df.head()

# %%
plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
sns.boxplot(x='Votes', data=df)

plt.subplot(2,2,2)
sns.histplot(df['Year'], color='g')

plt.subplot(2,2,3)
sns.histplot(df['Rating'], color='g')

plt.subplot(2,2,4)
sns.scatterplot(x=df['Duration'], y=df['Rating'], data=df)

plt.tight_layout()
plt.show()

# %%
df.hist(figsize=(30,15))
None

# %%
df.drop(['Name','Director','Actor 1','Actor 2','Actor 3'], axis=1,inplace=True)
df.head()



# %%
import numpy as np
import pandas as pd 
import seaborn
import matplotlib.pyplot as plt

# %%
#reading csv
file = ('Advertising.csv')

df = pd.read_csv(file)
df.head()

# %%
df.shape

# %%

x=df.iloc[:,0:-1]
x

# %%
df.describe()

# %%
y=df.iloc[:,-1]
y

# %%
import seaborn as sns
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7)

# %%
df['TV'].plot.hist(bins=10,xlabel="TV")

# %%
df['Radio'].plot.hist(bins=10,color='purple',xlabel="Radio")

# %%
df["Newspaper"].plot.hist(bins=10,color='green',xlabel="Newspaper")

# %%
sns.heatmap(df.corr(),annot = True)
plt.show

# %%
#train test split

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

# %%
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
x_train_scaled = Sc.fit_transform(x_train)

# %%

x_test_scaled=Sc.fit_transform(x_test)

# %%
#using regression model

# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train_scaled,y_train)

# %%

y_pred=reg.predict(x_test_scaled)

# %%
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

# %%
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,c='g')



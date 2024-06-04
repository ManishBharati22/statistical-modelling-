
#IMPORT LIBRAIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Read csv file 
df= pd.read_csv("musicdata.csv")
print(df.head(3))

##DATA EXPLORATION

print(df.info())
print(df.describe())
print(df.isnull().sum())
df =df.drop(columns=["Unnamed: 0"])
print(df)

#IMPUTING MISSING VALUES

missing_value = ["Track Name", "Artists", "Album Name"]
df[missing_value] = df[missing_value].fillna('uk')

print(df.isnull().sum())



##VISUALIZATION

plt.figure(figsize=(10, 6))
sns.histplot(df['Popularity'], bins=20, kde=True)
plt.title('Distribution of Popularity Scores')
plt.xlabel('Popularity Score')
plt.ylabel('Frequency')
plt.show()


"visualising distribution for understanding the relation between the features of data set"


numeric=df[['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Valence', 'Explicit', 'Key', 'Mode', 'Speechiness', 'Instrumentalness', 'Tempo']]
plt.figure(figsize=(5,5))
sns.pairplot(df)
plt.show()


#df['Explicit'] = df['Explicit'].astype(int)
print(df.info())




# preparing the dataset for regression
# convert 'Explicit' from boolean to integer (0 or 1)
df['Explicit'] = df['Explicit'].astype(int)

# selecting features and target for the model
features = ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Valence', 'Explicit', 'Key', 'Mode', 'Speechiness', 'Instrumentalness', 'Tempo']
X = df[features]
y = df['Popularity']

# standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predicting on the test set
y_pred = model.predict(X_test)

# evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# outputting the coefficients and performance metrics
coefficients = pd.Series(model.coef_, index=features)
print(coefficients)

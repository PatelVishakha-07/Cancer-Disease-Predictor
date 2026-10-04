import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("diagnosis.csv")

df.drop(["id", "Unnamed: 32"], inplace=True, axis=1)

x = df.iloc[:,1:]
y = df.iloc[:, [0]]

le = LabelEncoder()
y = le.fit_transform(y)
y = pd.DataFrame(y, columns=["diagnosis"])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)

x_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y)
y_predict = knn.predict(x_test_scaled)
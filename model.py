import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("diagnosis.csv")

df.drop(["id", "Unnamed: 32"], inplace=True, axis=1)

x = df.iloc[:,1:]
y = df.iloc[:, [0]]

le = LabelEncoder()
y = le.fit_transform(y)
y = pd.DataFrame(y, columns=["diagnosis"])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)

x_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)
y_predict = knn.predict(x_test_scaled)

ac_score = accuracy_score(y_test, y_predict)
print("Accuracy Score:", round(ac_score, 3))
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict, target_names=le.classes_))

with open("model.pkl", "wb") as f:
    pickle.dump({"scaler":scaler, "knn":knn, "features":list(x.columns), "classes":list(le.classes_),
                  "accuracy":float(ac_score)}, f)

print("model.pkl saved")

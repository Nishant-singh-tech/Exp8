
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv("Data\Iris.csv.csv")
df = pd.read_csv("Data/Iris.csv.csv")

df = df.drop("Id", axis=1)

X = df.drop("Species", axis=1)
Y = df["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

pred = model.predict(X_test)

acc = accuracy_score(Y_test, pred)
print("Accuracy:", acc)

joblib.dump(model, "model.pkl")

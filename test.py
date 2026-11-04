#check the model validity
#not train the model
#regerious testing
#assertions for software testing
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
test_df=pd.read_csv("Data\Iris.csv.csv")
test_df=test_df.drop("Id",axis=1)
X=test_df.drop("Species",axis=1)
Y = test_df["Species"]

pred = model.predict(X)
accu = accuracy_score(pred,Y)
print(accu)

assert accu>=0.97
assert len(pred)==len(Y)
print("All test cases passed and model is ready to deploy")
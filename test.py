#check the model validity
#not train the model
#regerious testing
#assertions for software testing
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
# <<<<<<< HEAD
# test_df=pd.read_csv("Data\Iris.csv.csv")

test_df=pd.read_csv("Data/Iris.csv.csv")
# >>>>>>> b706f59bfa35a1e15d6f58faf159656262809bf4
test_df=test_df.drop("Id",axis=1)
X=test_df.drop("Species",axis=1)
Y = test_df["Species"]

pred = model.predict(X)
accu = accuracy_score(pred,Y)
print(accu)

assert accu>=0.97
assert len(pred)==len(Y)
<<<<<<< HEAD
print("All test cases passed and model is ready to deploy")
=======
print("All test cases passed and model is ready to deploy")
>>>>>>> b706f59bfa35a1e15d6f58faf159656262809bf4

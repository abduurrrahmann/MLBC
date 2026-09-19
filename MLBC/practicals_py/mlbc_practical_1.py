import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

df = pd.read_csv(url, names=columns)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

model = LogisticRegression(max_iter=800)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy with all features:", accuracy_score(y_test, y_pred))

selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

selected_features = selector.get_support(indices=True)

print("Selected feature column indices:", selected_features)
print("Selected feature names:", [columns[i] for i in selected_features])

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new,
    y,
    test_size=0.2,
    random_state=1
)

model2 = LogisticRegression(max_iter=200)
model2.fit(X_train_new, y_train_new)

y_pred_new = model2.predict(X_test_new)

print("Accuracy with selected features:", accuracy_score(y_test_new, y_pred_new))
"""PS C:\Users\thele\Desktop\sem 7 submissions> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "c:\Users\thele\Desktop\sem 7 submissions\.venv\Scripts\Activate.ps1")
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> & "c:\Users\thele\Desktop\sem 7 submissions\.venv\Scripts\python.exe" "c:/Users/thele/Desktop/sem 7 submissions/MLBC/mlbc_practical_1.py"
Accuracy with all features: 0.7792207792207793
Selected feature column indices: [0 1 5 6 7]
Selected feature names: ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
Accuracy with selected features: 0.7662337662337663
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions>

PS C:\Users\thele\Desktop\sem 7 submissions> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "c:\Users\thele\Desktop\sem 7 submissions\.venv\Scripts\Activate.ps1")
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python--version
python--version : The term 'python--version' is not recognized as the name of a cmdlet, function, script file, or 
operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try 
again.
At line:1 char:1
+ python--version
+ ~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (python--version:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
 
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python --version
Python 3.11.9
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python -c "import sys; print(sys.executable)"
C:\Users\thele\Desktop\sem 7 submissions\.venv\Scripts\python.exe
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python -c "import pandas, sklearn; print('MLBC environment OK')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'pandas'
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python -m pip install pandas scikit-learn
Collecting pandas
  Downloading pandas-3.0.5-cp311-cp311-win_amd64.whl.metadata (19 kB)
Collecting scikit-learn
  Downloading scikit_learn-1.9.0-cp311-cp311-win_amd64.whl.metadata (11 kB)
Collecting numpy>=1.26.0 (from pandas)
  Downloading numpy-2.4.6-cp311-cp311-win_amd64.whl.metadata (6.6 kB)
Collecting python-dateutil>=2.8.2 (from pandas)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting tzdata (from pandas)
  Downloading tzdata-2026.3-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting scipy>=1.10.0 (from scikit-learn)
  Using cached scipy-1.17.1-cp311-cp311-win_amd64.whl.metadata (60 kB)
Collecting joblib>=1.4.0 (from scikit-learn)
  Using cached joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting narwhals>=2.0.1 (from scikit-learn)
  Downloading narwhals-2.25.0-py3-none-any.whl.metadata (15 kB)
Collecting threadpoolctl>=3.5.0 (from scikit-learn)
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Downloading pandas-3.0.5-cp311-cp311-win_amd64.whl (10.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.0/10.0 MB 1.5 MB/s  0:00:06
Downloading scikit_learn-1.9.0-cp311-cp311-win_amd64.whl (8.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/8.3 MB 1.3 MB/s  0:00:06
Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
Downloading narwhals-2.25.0-py3-none-any.whl (467 kB)
Downloading numpy-2.4.6-cp311-cp311-win_amd64.whl (12.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.6/12.6 MB 1.8 MB/s  0:00:07
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading scipy-1.17.1-cp311-cp311-win_amd64.whl (36.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.6/36.6 MB 740.0 kB/s  0:00:46
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading tzdata-2026.3-py2.py3-none-any.whl (348 kB)
Installing collected packages: tzdata, threadpoolctl, six, numpy, narwhals, joblib, scipy, python-dateutil, scikit-learn, pandas
Successfully installed joblib-1.5.3 narwhals-2.25.0 numpy-2.4.6 pandas-3.0.5 python-dateutil-2.9.0.post0 scikit-learn-1.9.0 scipy-1.17.1 six-1.17.0 threadpoolctl-3.6.0 tzdata-2026.3
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python -c "import pandas, sklearn; print('MLBC environment OK')"
MLBC environment OK
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> python -c "import pandas, sklearn; print('Pandas:',pandas.__version__); print('Scikit-learn:',sklearn.__version__)"
Pandas: 3.0.5
Scikit-learn: 1.9.0
(.venv) PS C:\Users\thele\Desktop\sem 7 submissions> 
"""
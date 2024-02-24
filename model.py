from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 訓練データとテストデータの読み込み
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 特徴量とターゲットの分離
y_train = train_data["Y"]
X_train = train_data.drop(["Y", "index"], axis=1)
X_test = test_data.drop(["index"], axis=1)

# カテゴリカルな特徴量と数値特徴量の識別
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
numerical_features = ["age", "fnlwgt", "education-num"]

# 数値データの前処理: 欠損値を中央値で補完し、標準化
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# カテゴリカルデータの前処理: 最頻値で欠損値を補完し、ワンホットエンコーディング
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

params_grid = [
    {
        "model": [RandomForestClassifier(random_state=0)],
        "model__n_estimators": [100, 300],
        "model__max_depth": [5, 10],
    },
    {
        "model": [LogisticRegression(random_state=0, max_iter=1000)],
        "model__C": [0.1, 1.0, 10.0],
    },
    {
        "model": [LGBMClassifier(random_state=0)],
        "model__n_estimators": [100, 300],
        "model__learning_rate": [0.05, 0.06, 0.01, 0.1],
    },
    {
        "model": [SVC(random_state=0)],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    },
]

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=0)),
    ]
)

grid_search = GridSearchCV(pipeline, params_grid, cv=5, verbose=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

# 最適なパラメータを表示
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

preds_test = grid_search.predict(X_test)

predictions_df = pd.DataFrame({"index": test_data["index"], "Y": preds_test})

output_file_name = "predictions.csv"
predictions_df.to_csv(output_file_name, index=False, header=False)

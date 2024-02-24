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
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)


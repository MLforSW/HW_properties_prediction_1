import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import joblib

# 读取CSV文件
df = pd.read_csv('NFPAtest.csv')  # 修改为你的CSV文件路径


# 定义一个函数将SMILES转换为Morgan指纹
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return [int(b) for b in fp.ToBitString()]


# 应用函数转换SMILES列
X = df['Canonical SMILES'].apply(smiles_to_fingerprint).tolist()
y = df.iloc[:, 4:5].values
X = np.array(X)
y = np.array(y).ravel()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 定义一个通用函数来训练和评估模型
def train_evaluate_model(model, param_space, X_train, y_train, X_test, y_test, model_name):
    bayes_cv = BayesSearchCV(model, param_space, n_iter=64, scoring='roc_auc', cv=3)
    bayes_cv.fit(X_train, y_train)
    print(f'Best parameters: {bayes_cv.best_params_}')

    best_model = bayes_cv.best_estimator_
    y_pred = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'The ROC-AUC score for {type(model).__name__} on the validation set is: {roc_auc}')

    # 保存模型
    model_path = f'{model_name}_best_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

# 定义一个函数来加载模型并进行预测
def load_model_and_predict(model_path, X_test):
    # 加载模型
    loaded_model = joblib.load(model_path)
    y_pred = loaded_model.predict_proba(X_test)[:, 1]
    return y_pred

# 定义搜索空间
svm_space = {'C': Real(1e-6, 1e+6, prior='log-uniform'),
             'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
             'kernel': Categorical(['linear', 'poly', 'rbf'])}

rf_space = {'n_estimators': Integer(10, 1000),
            'max_depth': Integer(1, 50),
            'min_samples_split': Real(0.01, 1.0, prior='log-uniform')}

xgb_space = {'n_estimators': Integer(10, 1000),
             'max_depth': Integer(1, 50),
             'learning_rate': Real(0.01, 1.0, prior='log-uniform')}

lgbm_space = {
    'n_estimators': Integer(10, 1000),
    'max_depth': Integer(1, 50),
    'num_leaves': Integer(2, 4096),  # 确保范围宽阔以探索不同的值
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
}

# 训练和评估每个模型
# train_evaluate_model(SVC(probability=True), svm_space, X_train, y_train, X_test, y_test, "SVM-W")
# train_evaluate_model(RandomForestClassifier(), rf_space, X_train, y_train, X_test, y_test, "Random-Forest-W")
# train_evaluate_model(XGBClassifier(eval_metric='mlogloss'), xgb_space, X_train, y_train, X_test, y_test, "XGBoost-W")
# train_evaluate_model(LGBMClassifier(verbose=-1), lgbm_space, X_train, y_train, X_test, y_test, "LightGBM-W")


# model_path = 'SVM_best_model.joblib'
# y_pred = load_model_and_predict(model_path, X_test)
# roc_auc = roc_auc_score(y_test, y_pred)
# print(roc_auc)
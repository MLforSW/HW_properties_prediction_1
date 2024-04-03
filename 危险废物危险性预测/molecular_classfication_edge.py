import pandas as pd
from keras.optimizer_experimental.adam import Adam
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import shap
from xgboost import XGBClassifier
import joblib
from sklearn.manifold import TSNE
# 读取CSV文件
df = pd.read_csv('NFPAtest.csv')  # 修改为你的CSV文件路径

# 定义一个函数将SMILES转换为Morgan指纹
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return [int(b) for b in fp.ToBitString()]

# 应用函数转换SMILES列
X = df['Canonical SMILES'].apply(smiles_to_fingerprint).tolist()
y_health = df.iloc[:, 1:2].values
y_fire = df.iloc[:, 2:3].values
y_reactivity = df.iloc[:, 3:4].values
y_Water = df.iloc[:, 4:5].values

# X = np.array(X)
# y_H = np.array(y_health).ravel()
# y_F = np.array(y_fire).ravel()
# y_R = np.array(y_reactivity)
# y_W = np.array(y_Water).ravel()

# 拆分数据集
X_H_train, X_H_test, y_H_train, y_H_test = train_test_split(X, y_health, test_size=0.2, random_state=0)
X_F_train, X_F_test, y_F_train, y_F_test = train_test_split(X, y_fire, test_size=0.2, random_state=0)
X_R_train, X_R_test, y_R_train, y_R_test = train_test_split(X, y_reactivity, test_size=0.2, random_state=0)
X_W_train, X_W_test, y_W_train, y_W_test = train_test_split(X, y_Water, test_size=0.2, random_state=0)


# 模型加载
H_model = joblib.load("XGBoost-H_best_model.joblib")
F_model = joblib.load("SVM-F_best_model.joblib")
R_model = load_model("best_model_reactivity.h5")
W_model = joblib.load("Random-Forest-W_best_model.joblib")

from sklearn.manifold import TSNE

# 定义一个函数来执行 t-SNE 降维和可视化，根据预测标签着色
def tsne_visualization_with_labels(model_name, model, X_test, y_test):
    # 获取预测标签
    if model_name == 'R_model':
        # 假设 R_model 是一个 Keras 模型
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    else:
        # 对于非 Keras 模型
        y_pred = model.predict(X_test)

    # 应用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=0)
    X_test_tsne = tsne.fit_transform(X_test)

    # 可视化，根据预测标签着色
    plt.figure(figsize=(8, 5))
    for class_value in np.unique(y_pred):
        # 根据类别值选择样本
        class_mask = y_pred == class_value
        plt.scatter(X_test_tsne[class_mask, 0], X_test_tsne[class_mask, 1], label=f'Class {class_value}')

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title(f't-SNE visualization of {model_name} predictions')
    plt.legend()
    plt.show()

# 对每个模型分别进行 t-SNE 可视化
tsne_visualization_with_labels('H_model', H_model, X_H_train, y_H_train)
tsne_visualization_with_labels('F_model', F_model, X_F_train, y_F_train)
tsne_visualization_with_labels('R_model', R_model, X_R_train, y_R_train)
tsne_visualization_with_labels('W_model', W_model, X_W_train, y_W_train)

import umap

def umap_visualization(model_name, model, X_test, y_test):
    # 获取预测标签
    if model_name == 'R_model':
        y_pred = (model.predict(X_test) >= 0.5).astype('int32').ravel()
    else:
        y_pred = model.predict(X_test)
        # 确保预测标签为整型，特别是对于非 Keras 模型
        y_pred = y_pred.astype('int32')

    # 打印以确认 y_pred 只包含 0 和 1
    print(np.unique(y_pred))

    # 应用 UMAP 降维
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_test)

    # 可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, cmap='coolwarm')
    plt.title(f'{model_name} Predictions Visualization with UMAP')
    plt.xlabel('UMAP dimension 1')
    plt.ylabel('UMAP dimension 2')
    plt.colorbar(scatter)
    plt.show()

# 对每个模型分别进行 UMAP 可视化
umap_visualization('XGBoost in Health', H_model, X_H_train, y_H_train)
umap_visualization('SVM in Fire', F_model, X_F_train, y_F_train)
umap_visualization('MLP in Reactivity', R_model, X_R_train, y_R_train)
umap_visualization('Random forest in W', W_model, X_W_train, y_W_train)


import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import umap
from keras.models import Model
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

X = np.array(X)
y_H = np.array(y_health)
y_F = np.array(y_fire)
y_R = np.array(y_reactivity)
y_W = np.array(y_Water)

# 拆分数据集
X_H_train, X_H_test, y_H_train, y_H_test = train_test_split(X, y_H, test_size=0.2, random_state=0)
X_F_train, X_F_test, y_F_train, y_F_test = train_test_split(X, y_F, test_size=0.2, random_state=0)
X_R_train, X_R_test, y_R_train, y_R_test = train_test_split(X, y_R, test_size=0.2, random_state=0)
X_W_train, X_W_test, y_W_train, y_W_test = train_test_split(X, y_W, test_size=0.2, random_state=0)

# 模型加载
Health = load_model("best_model.h5")
Fire = load_model("best_model_fire.h5")
Reactivity = load_model("best_model_reactivity.h5")
Water = load_model("best_model_W.h5")

# 定义字号
axis_label_fontsize = 18
tick_label_fontsize = 15
colorbar_tick_label_fontsize = 15  # 设置颜色条刻度标签的字号

# # 遍历模型的每一层
# for i, layer in enumerate(Water.layers):
#     weights = layer.get_weights()
#     if weights:  # 检查层是否有权重
#         weight_matrix = weights[0]
#         normalized_weights = (weight_matrix - np.min(weight_matrix)) / (np.max(weight_matrix) - np.min(weight_matrix))
#
#         plt.figure(figsize=(10, 8))
#         cax = plt.matshow(normalized_weights, cmap='RdBu', fignum=1, aspect='auto')
#         cbar = plt.colorbar(cax)
#
#         # 增大颜色条刻度标签的字号
#         cbar.ax.tick_params(labelsize=colorbar_tick_label_fontsize)
#
#         # 设置坐标轴标题和刻度标签的字号
#         plt.title(f'Layer {i} Normalized Weights', fontsize=axis_label_fontsize)
#         plt.xlabel('Output Neurons', fontsize=axis_label_fontsize)
#         plt.ylabel('Input Neurons', fontsize=axis_label_fontsize)
#
#         # 设置刻度标签的字号
#         plt.xticks(fontsize=tick_label_fontsize)
#         plt.yticks(fontsize=tick_label_fontsize)
#
#         plt.grid(True)
#         plt.show()

def process_and_visualize(model, X_train, y_train, model_name):
    # 确保y_train是一维数组，适用于二元分类
    if y_train.ndim > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()

    # 创建一个新模型，该模型包含原始模型的前三层
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[2].output)

    # 使用这个新模型获取数据的中间层输出
    intermediate_output = intermediate_layer_model.predict(X_train)

    # 使用 UMAP 进行降维
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(intermediate_output)

    # 可视化，使用y_train作为标签
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_train, cmap='coolwarm')
    plt.colorbar(scatter)
    plt.title(f'UMAP visualization of {model_name} with training labels')
    plt.xlabel('UMAP dimension 1')
    plt.ylabel('UMAP dimension 2')
    plt.show()


# 对四个模型分别进行处理和可视化
process_and_visualize(Health, X_H_train, y_H_train, "Health")
process_and_visualize(Fire, X_F_train, y_F_train, "Fire")
process_and_visualize(Reactivity, X_R_train, y_R_train, "Reactivity")
process_and_visualize(Water, X_W_train, y_W_train, "W")
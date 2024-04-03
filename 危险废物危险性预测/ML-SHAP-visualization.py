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
import shap
import seaborn as sns

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
y_H = np.array(y_health).ravel()
y_F = np.array(y_fire).ravel()
y_R = np.array(y_reactivity).ravel()
y_W = np.array(y_Water).ravel()

# 拆分数据集
X_H_train, X_H_test, y_H_train, y_H_test = train_test_split(X, y_H, test_size=0.2, random_state=0)
X_F_train, X_F_test, y_F_train, y_F_test = train_test_split(X, y_F, test_size=0.2, random_state=0)
X_R_train, X_R_test, y_R_train, y_R_test = train_test_split(X, y_R, test_size=0.2, random_state=0)
X_W_train, X_W_test, y_W_train, y_W_test = train_test_split(X, y_H, test_size=0.2, random_state=0)

# XGBoost模型加载
XGBoost_H_model = joblib.load("XGBoost-H_best_model.joblib")
XGBoost_F_model = joblib.load("XGBoost-F_best_model.joblib")
XGBoost_R_model = joblib.load("XGBoost-R_best_model.joblib")
XGBoost_W_model = joblib.load("XGBoost-W_best_model.joblib")

# SVM模型加载
SVM_H_model = joblib.load("SVM-H_best_model.joblib")
SVM_F_model = joblib.load("SVM-F_best_model.joblib")
SVM_R_model = joblib.load("SVM-R_best_model.joblib")
SVM_W_model = joblib.load("SVM-W_best_model.joblib")

# RF模型加载
RF_H_model = joblib.load("Random-Forest-H_best_model.joblib")
RF_F_model = joblib.load("Random-Forest-F_best_model.joblib")
RF_R_model = joblib.load("Random-Forest-R_best_model.joblib")
RF_W_model = joblib.load("Random-Forest-W_best_model.joblib")

# LightGBM模型加载
LightGBM_H_model = joblib.load("LightGBM-H_best_model.joblib")
LightGBM_F_model = joblib.load("LightGBM-F_best_model.joblib")
LightGBM_R_model = joblib.load("LightGBM-R_best_model.joblib")
LightGBM_W_model = joblib.load("LightGBM-W_best_model.joblib")

# MLP模型加载
MLP_H_model = load_model("best_model.h5")
MLP_F_model = load_model("best_model_fire.h5")
MLP_R_model = load_model("best_model_reactivity.h5")
MLP_W_model = load_model("best_model_W.h5")


XGBoost_models = [XGBoost_H_model, XGBoost_F_model, XGBoost_R_model, XGBoost_W_model]
SVM_models = [SVM_H_model, SVM_F_model, SVM_R_model, SVM_W_model]
RF_models = [RF_H_model, RF_F_model, RF_R_model, RF_W_model]
LightGBM_models = [LightGBM_H_model, LightGBM_F_model, LightGBM_R_model, LightGBM_W_model]
MLP_models = [MLP_H_model, MLP_F_model, MLP_R_model, MLP_W_model]

XGBoost_model_names = ['XGBoost_H_model', 'XGBoost_F_model', 'XGBoost_R_model', 'XGBoost_W_model']
SVM_model_names = ['SVM_H_model', 'SVM_F_model', 'SVM_R_model', 'SVM_W_model']
RF_model_names = ['RF_H_model', 'RF_F_model', 'RF_R_model', 'RF_W_model']
LightGBM_model_names = ['LightGBM_H_model', 'lightGBM_F_model', 'LightGBM_R_model', 'LightGBM_W_model']
MLP_model_names = ['MLP_H_model', 'MLP_F_model', 'MLP_R_model', 'MLP_W_model']

X_train_sets = [X_H_train, X_F_train, X_R_train, X_W_train]

# 初始化一个空列表来存储每个模型的平均 SHAP 值向量
XGBoost_average_shap_values = []
SVM_average_shap_values = []
RF_average_shap_values = []
LightGBM_average_shap_values = []
MLP_average_shap_values = []

# 对每个模型计算 SHAP 值并获取平均值
for model, X_train in zip(XGBoost_models, X_train_sets):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    # 先取绝对值再计算所有样本上每个特征的平均 SHAP 值
    average_shap = np.mean(np.abs(shap_values), axis=0)
    XGBoost_average_shap_values.append(average_shap)

for model, X_train in zip(SVM_models, X_train_sets):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 30))
    shap_values = explainer.shap_values(X_train)
    # 先取绝对值再计算所有样本上每个特征的平均 SHAP 值
    average_shap = np.mean(np.abs(shap_values), axis=0)
    SVM_average_shap_values.append(average_shap)

for model, X_train in zip(RF_models, X_train_sets):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 30))
    shap_values = explainer.shap_values(X_train)
    # 先取绝对值再计算所有样本上每个特征的平均 SHAP 值
    average_shap = np.mean(np.abs(shap_values), axis=0)
    RF_average_shap_values.append(average_shap)

for model, X_train in zip(LightGBM_models, X_train_sets):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 30))
    shap_values = explainer.shap_values(X_train)
    # 先取绝对值再计算所有样本上每个特征的平均 SHAP 值
    average_shap = np.mean(np.abs(shap_values), axis=0)
    LightGBM_average_shap_values.append(average_shap)

for model, X_train in zip(MLP_models, X_train_sets):
    explainer = shap.DeepExplainer(model, shap.sample(X_train, 30))
    shap_values = explainer.shap_values(X_train)
    # 先取绝对值再计算所有样本上每个特征的平均 SHAP 值
    average_shap = np.mean(np.abs(shap_values), axis=0)
    MLP_average_shap_values.append(average_shap)

# 将平均SHAP值列表转换为NumPy数组
XGBoost_avg_shap_array = np.array(XGBoost_average_shap_values)
SVM_avg_shap_array = np.array(SVM_average_shap_values)
RF_avg_shap_array = np.array(RF_average_shap_values)
LightGBM_avg_shap_array = np.array(LightGBM_average_shap_values)
MLP_avg_shap_array = np.array(MLP_average_shap_values)

def check_and_adjust_shap_dimensions(shap_values_array):
    if shap_values_array.ndim > 2:
        # 假设额外的维度是类别维度，取平均以移除类别维度
        return shap_values_array.mean(axis=1)
    return shap_values_array

XGBoost_avg_shap_array = check_and_adjust_shap_dimensions(XGBoost_avg_shap_array)
SVM_avg_shap_array = check_and_adjust_shap_dimensions(SVM_avg_shap_array)
RF_avg_shap_array = check_and_adjust_shap_dimensions(RF_avg_shap_array)
LightGBM_avg_shap_array = check_and_adjust_shap_dimensions(LightGBM_avg_shap_array)
MLP_avg_shap_array = check_and_adjust_shap_dimensions(MLP_avg_shap_array)

# 将所有模型的平均SHAP值数组垂直堆叠起来
combined_shap_values = np.vstack((XGBoost_avg_shap_array, SVM_avg_shap_array, RF_avg_shap_array, LightGBM_avg_shap_array, MLP_avg_shap_array))

# 合并所有模型名称
model_names = XGBoost_model_names + SVM_model_names + RF_model_names + LightGBM_model_names + MLP_model_names

# 创建一个空的DataFrame，用于存储模型名称和平均SHAP值
shap_values_df = pd.DataFrame()

# 将模型名称作为第一列添加到DataFrame中
shap_values_df['Model'] = model_names

# 为每个模型的SHAP值创建列名
feature_columns = [f'Feature_{i}' for i in range(XGBoost_avg_shap_array.shape[1])]

# 将所有模型的SHAP值添加到DataFrame中
for i, model_name in enumerate(model_names):
    shap_values_df.loc[shap_values_df['Model'] == model_name, feature_columns] = combined_shap_values[i]

# 保存DataFrame为CSV文件
shap_values_df.to_csv("shap_new.csv", index=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV文件
shap_values_df = pd.read_csv("shap_new.csv")

# 确保模型名称是DataFrame的索引
shap_values_df.set_index('Model', inplace=True)

# 转换为浮点数以确保可以绘制热力图
shap_values_df = shap_values_df.astype(float)

# 创建掩码以只填充非零值
mask = shap_values_df == 0

# 使用单调色系映射，从浅到深反映SHAP值的增加
cmap = sns.light_palette("navy", as_cmap=True)

# 绘制热力图
plt.figure(figsize=(14, 12))
ax = sns.heatmap(shap_values_df, cmap=cmap, annot=False, mask=mask, cbar_kws={'label': 'SHAP Value'})

# 调整颜色条（color bar）标签的字体大小
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('SHAP Value', fontsize=14)  # 设置颜色条标题的字体大小
cbar.ax.tick_params(labelsize=12)  # 设置颜色条刻度标签的字体大小

# 设置横坐标每隔256个显示一个标签，并格式化为feature-编号
feature_labels = [f'Feature-{i}' for i in range(1, shap_values_df.shape[1]+1, 256)]
ax.set_xticks(np.arange(0, shap_values_df.shape[1], 256) + 0.5)  # 将标签移动到区块的中心
ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=14)

# 去掉横坐标下的features字样
plt.xlabel('')
plt.title('SHAP Absolute Values Heatmap', fontsize=20)
plt.ylabel('Models', fontsize=14)
plt.yticks(fontsize=14)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
shap_values_df = pd.read_csv("shap_new.csv", index_col='Model')

# Convert the DataFrame to float to ensure heatmap can be plotted
shap_values_df = shap_values_df.astype(float)

# Task identifiers
tasks = ['H', 'F', 'R', 'W']
task_names = ['Health', 'Fire', 'Reactivity', 'W']

# Initialize a figure for the plots
plt.figure(figsize=(24, 24))

for i, task in enumerate(tasks):
    # Filter models by task
    task_models = shap_values_df.filter(regex=f'_{task}_', axis=0)

    # Calculate the average of the absolute SHAP values for each feature for the task
    average_abs_shap_values_task = task_models.abs().mean(axis=0)

    # Sort the features based on their average absolute SHAP value and select the top 10
    top_10_features_task = average_abs_shap_values_task.sort_values(ascending=False).head(10)

    # Plot
    plt.subplot(2, 2, i + 1)
    sns.barplot(x=top_10_features_task.values, y=top_10_features_task.index, color="skyblue")
    plt.title(f'Top 10 Important Features for {task_names[i]} Task Based on SHAP Values', fontsize=16)
    plt.xlabel('Average Absolute SHAP Value', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

# wspace和hspace控制宽度和高度方向上的子图间距
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)

plt.show()
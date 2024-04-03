import pandas as pd
from keras.optimizer_experimental.adam import Adam
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from keras.callbacks import Callback
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('NFPAtest.csv')  # 修改为你的CSV文件路径


# 定义将SMILES转换为Morgan指纹的函数
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return [int(b) for b in fp.ToBitString()]


# 应用函数转换SMILES列
X = df['Canonical SMILES'].apply(smiles_to_fingerprint).tolist()
y = df.iloc[:, 4:5].values  # 假设输出列是从第二列开始的
X = np.array(X)
y = np.array(y)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 定义回调函数
class PrintLogs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1}: Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Val_Loss: {logs['val_loss']}, Val_Accuracy: {logs['val_accuracy']}")


# 定义超参数空间
space = [
    Integer(128, 256, name='n_hidden_1'),
    Integer(64, 128, name='n_hidden_2'),
    Integer(32, 64, name='n_hidden_3'),
    Real(10 ** -3, 10 ** -2, "log-uniform", name='learning_rate')
]


# 定义目标函数
@use_named_args(space)
def objective(**params):
    print(f"Training with params: {params}")

    model = Sequential([
        Dense(params['n_hidden_1'], activation='relu', input_shape=(X.shape[1],)),
        Dense(params['n_hidden_2'], activation='relu'),
        Dense(params['n_hidden_3'], activation='relu'),
        Dense(y_train.shape[1], activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0,
              callbacks=[PrintLogs()])

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
    return loss


# 执行贝叶斯优化
res = gp_minimize(objective, space, n_calls=100, random_state=0, verbose=True)

print("Best parameters: {}".format(res.x))
print("Best MSE: {:.4f}".format(res.fun))

# 绘制收敛图
plot_convergence(res)
plt.show()

# 绘制每个维度的评估结果
plot_evaluations(res, bins=10)
plt.show()

# 绘制目标空间
plot_objective(res)
plt.show()
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from urllib.request import urlopen
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import joblib
from keras.models import load_model

# 确保导入了所有必要的库和之前定义的函数
# 函数：根据CAS号查询SMILES字符串
def fetch_smiles(cas):
    try:
        url = f'http://cactus.nci.nih.gov/chemical/structure/{cas}/smiles'
        response = urlopen(url)
        smiles = response.read().decode('utf8')
        return smiles
    except Exception as e:
        print(f"Error fetching SMILES for {cas}: {e}")
        return None

# 定义将SMILES转换为Morgan指纹的函数
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array([int(b) for b in fp.ToBitString()])


def is_organic(smiles):
    """判断给定的SMILES字符串是否表示一个有机化合物，基于更准确的标准"""
    # 定义允许的元素集合
    allowed_atoms = {'C', 'H', 'O', 'S', 'N', 'P', 'Br', 'Cl', 'I', 'Si'}
    # 定义排除的特定SMILES
    excluded_smiles = {"[C]", "CO", "C(=O)=O", "C(=O)(O)O", "C(=O)([O-])[O-].[NH4+].[NH4+]"}

    # 如果SMILES在排除列表中，直接返回False
    if smiles in excluded_smiles:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    # 检查是否含有碳原子
    has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
    # 检查所有原子是否都在允许的元素集合内
    all_atoms_allowed = all(atom.GetSymbol() in allowed_atoms for atom in mol.GetAtoms())

    return has_carbon and all_atoms_allowed


def search_inorganic_properties(smiles, database="NFPA无机数据库.csv"):
    """从无机化合物数据库中搜索给定SMILES字符串的属性"""
    try:
        df = pd.read_csv(database)
        row = df[df['SMILES'] == smiles]
        if not row.empty:
            return row.iloc[0][['Health', 'Fire', 'Reactivity', 'W']].to_dict()
        else:
            return None
    except FileNotFoundError:
        print(f"Database file {database} not found.")
        return None

class HazardPredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chemical Hazard Prediction")
        self.geometry("800x400")

        # 模型加载
        self.load_models()

        # 输入框和按钮
        self.label = tk.Label(self, text="Enter CAS Numbers (comma separated):")
        self.label.pack(pady=10)

        self.entry = tk.Entry(self, width=50)
        self.entry.pack(pady=10)

        self.predict_button = tk.Button(self, text="Predict", command=self.predict_and_show)
        self.predict_button.pack(pady=20)

    def load_models(self):
        # 加载模型
        self.health = joblib.load("XGBoost-H_best_model.joblib")
        self.fire = joblib.load("SVM-F_best_model.joblib")
        self.reactivity = load_model("best_model_reactivity.h5")
        self.water = joblib.load("Random-Forest-W_best_model.joblib")

    def predict_and_show(self):
        cas_input = self.entry.get()
        if not cas_input:
            messagebox.showinfo("Error", "Please enter at least one CAS number.")
            return
        cas_numbers = [cas.strip() for cas in cas_input.split(",")]

        properties_list = self.process_cas_numbers(cas_numbers)

        # 显示结果的新窗口
        self.show_results(properties_list)

    def process_cas_numbers(self, cas_numbers):
        properties_list = []
        mix_properties = {'Health': 0, 'Fire': 0, 'Reactivity': 0, 'Water': 0}

        for cas in cas_numbers:
            smiles = fetch_smiles(cas)
            if not smiles:
                print(f"CAS号: {cas} 没有找到有效的SMILES字符串。")
                continue

            if is_organic(smiles):
                fingerprint = smiles_to_fingerprint(smiles)
                h_pred = self.health.predict([fingerprint])[0]
                f_pred = self.fire.predict([fingerprint])[0]
                r_pred = self.reactivity.predict(fingerprint.reshape(1, -1)).squeeze()
                w_pred = self.water.predict([fingerprint])[0]
                r_pred_binary = '1' if r_pred > 0.5 else '0'
                current_props = {'CAS': cas, 'Health': int(h_pred), 'Fire': int(f_pred),
                                 'Reactivity': int(r_pred_binary), 'Water': int(w_pred)}
            else:
                properties = search_inorganic_properties(smiles)
                if properties:
                    current_props = {'CAS': cas, 'Health': int(properties['Health']), 'Fire': int(properties['Fire']),
                                     'Reactivity': int(properties['Reactivity']), 'Water': int(properties.get('W', 0))}
                else:
                    current_props = {'CAS': cas, 'Health': 'N/A', 'Fire': 'N/A', 'Reactivity': 'N/A', 'Water': 'N/A'}

            properties_list.append(current_props)

            # 更新混合体系的特性
            for key in mix_properties.keys():
                mix_properties[key] |= int(current_props[key]) if current_props[key] != 'N/A' else 0

        # 添加混合体系的特性到列表
        properties_list.append(
            {'CAS': 'Mixed System', 'Health': mix_properties['Health'], 'Fire': mix_properties['Fire'],
             'Reactivity': mix_properties['Reactivity'], 'Water': mix_properties['Water']})
        return properties_list

    def show_results(self, properties_list):
        result_window = tk.Toplevel(self)
        result_window.title("Prediction Results")

        frame = ttk.Frame(result_window)
        frame.pack(fill='both', expand=True)

        # 使用pandas DataFrame来格式化数据
        df = pd.DataFrame(properties_list)

        # 创建表格展示结果
        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
        for column in df.columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="center")

        for _, row in df.iterrows():
            tree.insert("", tk.END, values=list(row))

        tree.pack(side='left', fill='both', expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')


if __name__ == "__main__":
    app = HazardPredictionApp()
    app.mainloop()

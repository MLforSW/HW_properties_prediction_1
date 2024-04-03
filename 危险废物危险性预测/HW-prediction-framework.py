import joblib
import numpy as np
from keras.models import load_model
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from urllib.request import urlopen


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


def main():
    # 加载模型
    health = joblib.load("XGBoost-H_best_model.joblib")
    fire = joblib.load("SVM-F_best_model.joblib")
    reactivity = load_model("best_model_reactivity.h5")
    water = joblib.load("Random-Forest-W_best_model.joblib")

    cas_input = input("请输入化学物质CAS号（如果有多个，请用逗号分隔）: ")
    cas_numbers = [cas.strip() for cas in cas_input.split(",")]

    # 用于存储每个化合物的特性信息
    properties_list = []

    mix_properties = {'Health': 0, 'Fire': 0, 'Reactivity': 0, 'Water': 0}

    for cas in cas_numbers:
        smiles = fetch_smiles(cas)
        if not smiles:
            print(f"CAS号: {cas} 没有找到有效的SMILES字符串。")
            continue

        if is_organic(smiles):
            fingerprint = smiles_to_fingerprint(smiles)
            h_pred = health.predict([fingerprint])[0]
            f_pred = fire.predict([fingerprint])[0]
            r_pred = reactivity.predict(fingerprint.reshape(1, -1)).squeeze()
            w_pred = water.predict([fingerprint])[0]
            r_pred_binary = '1' if r_pred > 0.5 else '0'

            current_props = {'CAS': cas, 'Health': h_pred, 'Fire': f_pred, 'Reactivity': r_pred_binary, 'Water': w_pred}
        else:
            properties = search_inorganic_properties(smiles)
            if properties:
                current_props = {'CAS': cas, 'Health': properties['Health'], 'Fire': properties['Fire'],
                                 'Reactivity': properties['Reactivity'], 'Water': properties.get('W', 0)}
            else:
                print(f"CAS号: {cas} 未查询到危险特性。")
                current_props = {'CAS': cas, 'Health': 'N/A', 'Fire': 'N/A', 'Reactivity': 'N/A', 'Water': 'N/A'}

        properties_list.append(current_props)

        # 更新混合体系的特性
        mix_properties['Health'] |= int(current_props['Health'])
        mix_properties['Fire'] |= int(current_props['Fire'])
        mix_properties['Reactivity'] |= int(current_props['Reactivity'])
        mix_properties['Water'] |= int(current_props['Water'])

    # 添加混合体系的特性到列表
    properties_list.append({'CAS': '混合体系', 'Health': mix_properties['Health'], 'Fire': mix_properties['Fire'],
                            'Reactivity': mix_properties['Reactivity'], 'Water': mix_properties['Water']})

    # 使用pandas展示结果
    df = pd.DataFrame(properties_list)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
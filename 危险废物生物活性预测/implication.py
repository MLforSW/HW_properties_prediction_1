import tkinter as tk
from tkinter import scrolledtext
from dgllife.model import AttentiveFPPredictor
import torch
from rdkit import Chem
from urllib.request import urlopen
import dgllife

# 设置原子和键的特征提取器
atom_featurizer = dgllife.utils.CanonicalAtomFeaturizer()
bond_featurizer = dgllife.utils.CanonicalBondFeaturizer()

Single_Attentive_FP = AttentiveFPPredictor(node_feat_size=74, edge_feat_size=12, num_layers=2, num_timesteps=2, n_tasks=1)
Multi_Attentive_FP = AttentiveFPPredictor(node_feat_size=74, edge_feat_size=12, num_layers=2, num_timesteps=2, n_tasks=5)

# 加载模型
Single_Attentive_FP.load_state_dict(torch.load('single_best_model_AttentiveFPPredictor.pth'))
Multi_Attentive_FP.load_state_dict(torch.load('multi_best_model_AttentiveFPPredictor.pth'))
Single_Attentive_FP.eval()
Multi_Attentive_FP.eval()

def fetch_smiles(cas):
    try:
        url = f'http://cactus.nci.nih.gov/chemical/structure/{cas}/smiles'
        response = urlopen(url)
        # 确保使用正确的编码来解码内容
        smiles = response.read().decode('utf-8')  # 明确使用 'utf-8' 编码
        return smiles.strip()  # 去除可能的前导和后缀空白字符
    except Exception as e:
        print(f"Error fetching SMILES for {cas}: {e}")
        return None

def predict(input_str):
    input_list = [item.strip() for item in input_str.split(',')]
    smiles_list = []

    # 获取 SMILES 字符串
    for item in input_list:
        if Chem.MolFromSmiles(item) is not None:
            smiles_list.append(item)
        else:
            smiles = fetch_smiles(item)
            if smiles is not None:
                smiles_list.append(smiles)

    # 初始化每个性质的计数器
    properties_count = [0] * 6

    # 定义标题和每列的宽度
    headers = ["SMILES", "General", "D-Tox", "G-Tox", "GPCR", "NR", "SR"]
    col_widths = [30, 8, 8, 8, 8, 8, 8]

    # 格式化标题行
    header_line = " ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
    output_lines = [header_line]

    # 对 SMILES 列表中的每个项目进行预测
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        bg = dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
        n_feats = bg.ndata.pop('h')
        e_feats = bg.edata.pop('e')

        # 获取预测结果
        general_logits = Single_Attentive_FP(bg, n_feats, e_feats)
        general_probs = torch.sigmoid(general_logits).squeeze(1)
        general_binary = general_probs.cpu().detach().numpy().round().astype(int).tolist()

        target_logits = Multi_Attentive_FP(bg, n_feats, e_feats)
        target_probs = torch.sigmoid(target_logits)
        target_binary = target_probs.cpu().detach().numpy().round().astype(int).tolist()

        # 更新计数器
        properties_count[0] += general_binary[0]
        for i, value in enumerate(target_binary[0]):
            properties_count[i+1] += value

        # 格式化每行结果
        line = [smiles.ljust(col_widths[0])]
        line.append(str(general_binary[0]).ljust(col_widths[1]))
        line += [str(value).ljust(col_widths[i+2]) for i, value in enumerate(target_binary[0])]
        formatted_line = " ".join(line)
        output_lines.append(formatted_line)

    # 添加 Mix system 行
    mix_system_values = [1 if count > 0 else 0 for count in properties_count]
    mix_system_line = ["Mix system".ljust(col_widths[0])]
    mix_system_line += [str(value).ljust(col_widths[i+1]) for i, value in enumerate(mix_system_values)]
    formatted_mix_system_line = " ".join(mix_system_line)
    output_lines.append(formatted_mix_system_line)

    # 将所有行合并为最终输出字符串
    return "\n".join(output_lines)

# 创建 Tkinter 主窗口
root = tk.Tk()
root.title("Hazard Waste Bio-effects Predictor")

# 创建输入框和输出框
input_label = tk.Label(root, text="Enter CAS numbers of hazard compounds(separated by commas):")
input_label.pack()

input_entry = tk.Entry(root, width=50)
input_entry.pack()

output_label = tk.Label(root, text="Output:")
output_label.pack()

# 设置新罗马字体（Times New Roman）并增加列宽
output_text = scrolledtext.ScrolledText(root, font=("Courier", 12), width=120, height=20)
output_text.pack()

# 创建预测按钮
def predict_button_clicked():
    input_str = input_entry.get()
    result = predict(input_str)
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, result)

predict_button = tk.Button(root, text="Predict", command=predict_button_clicked)
predict_button.pack()

# 运行 Tkinter 主循环
root.mainloop()
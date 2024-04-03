from dgllife.model import AttentiveFPPredictor
import torch
from rdkit import Chem
from urllib.request import urlopen
import dgllife
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from IPython.display import display
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import HTML

# 设置原子和键的特征提取器
atom_featurizer = dgllife.utils.CanonicalAtomFeaturizer()
bond_featurizer = dgllife.utils.CanonicalBondFeaturizer()

Single_Attentive_FP = AttentiveFPPredictor(node_feat_size=74, edge_feat_size=12, num_layers=2, num_timesteps=2,
                                           n_tasks=1)
Multi_Attentive_FP = AttentiveFPPredictor(node_feat_size=74, edge_feat_size=12, num_layers=2, num_timesteps=2,
                                          n_tasks=5)

# 加载模型
Single_Attentive_FP.load_state_dict(torch.load(
    r'D:\pycharm\pycharmprojects\Chapter3-Bio-effects-of-organic-compounds\single_best_model_AttentiveFPPredictor.pth'))
Multi_Attentive_FP.load_state_dict(torch.load(
    r'D:\pycharm\pycharmprojects\Chapter3-Bio-effects-of-organic-compounds\multi_best_model_AttentiveFPPredictor.pth'))
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


def predict(input_list):
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
            properties_count[i + 1] += value

        # 格式化每行结果
        line = [smiles.ljust(col_widths[0])]
        line.append(str(general_binary[0]).ljust(col_widths[1]))
        line += [str(value).ljust(col_widths[i + 2]) for i, value in enumerate(target_binary[0])]
        formatted_line = " ".join(line)
        output_lines.append(formatted_line)

    # 添加 Mix system 行
    mix_system_values = [1 if count > 0 else 0 for count in properties_count]
    mix_system_line = ["Mix system".ljust(col_widths[0])]
    mix_system_line += [str(value).ljust(col_widths[i + 1]) for i, value in enumerate(mix_system_values)]
    formatted_mix_system_line = " ".join(mix_system_line)
    output_lines.append(formatted_mix_system_line)

    # 将所有行合并为最终输出字符串
    return "\n".join(output_lines)


def drawmol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bg = dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    atom_feats = bg.ndata.pop('h')
    bond_feats = bg.edata.pop('e')
    _, atom_weights_single = Single_Attentive_FP(bg, atom_feats, bond_feats, get_node_weight=True)
    assert 0 < len(atom_weights_single)
    atom_weights_single = atom_weights_single[0]
    min_value = torch.min(atom_weights_single)
    max_value = torch.max(atom_weights_single)
    atom_weights_single = (atom_weights_single - min_value) / (max_value - min_value)

    _, atom_weights_multi = Multi_Attentive_FP(bg, atom_feats, bond_feats, get_node_weight=True)
    assert 0 < len(atom_weights_multi)
    atom_weights_multi = atom_weights_multi[0]
    min_value = torch.min(atom_weights_multi)
    max_value = torch.max(atom_weights_multi)
    atom_weights_multi = (atom_weights_multi - min_value) / (max_value - min_value)

    def normalize_and_color(weights):
        min_value = torch.min(weights)
        max_value = torch.max(weights)
        weights = (weights - min_value) / (max_value - min_value)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('bwr')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {i: plt_colors.to_rgba(weights[i].data.item()) for i in range(bg.number_of_nodes())}
        return atom_colors

    atom_colors_single = normalize_and_color(atom_weights_single)
    atom_colors_multi = normalize_and_color(atom_weights_multi)

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)

    # 准备画布
    def draw_molecule_with_weights(colors):
        drawer = rdMolDraw2D.MolDraw2DSVG(180, 180)
        drawer.SetFontSize(1)
        drawer.DrawMolecule(mol, highlightAtoms=range(bg.number_of_nodes()),
                            highlightBonds=[],
                            highlightAtomColors=colors)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace('svg:', '')
        return svg

    # 绘制使用Single模型的分子
    svg_single = draw_molecule_with_weights(atom_colors_single)
    # 绘制使用Multi模型的分子
    svg_multi = draw_molecule_with_weights(atom_colors_multi)

    # 返回结果
    return (smiles, svg_single, svg_multi)


def get_cas_from_user():
    user_input = input("Please enter SMILES or CAS numbers separated by commas: ")
    input_list = [item.strip() for item in user_input.split(',')]
    return input_list


user_input = get_cas_from_user()
smiles_list = []
for item in user_input:
    if Chem.MolFromSmiles(item) is not None:
        smiles_list.append(item)
    else:
        smiles = fetch_smiles(item)
        if smiles is not None:
            smiles_list.append(smiles)

predict_result = predict(user_input)
print(predict_result)
# 初始化HTML表格
table_html = '<table border="1" style="font-size:16px;">'
table_html += '<tr><th>CAS/SMILES</th><th>Single-visual</th><th>Multi-visual</th></tr>'

for index, smiles in enumerate(smiles_list):
    cas_number = user_input[index]  # 从input_list获取对应的CAS号
    smiles, svg_single, svg_multi = drawmol(smiles)  # 生成对应的SVG图像

    # 创建HTML表格行
    table_html += f'<tr>'
    table_html += f'<td style="text-align:center">{cas_number}<br/>{smiles}</td>'
    table_html += f'<td style="text-align:center">{SVG(svg_single)._repr_svg_()}</td>'
    table_html += f'<td style="text-align:center">{SVG(svg_multi)._repr_svg_()}</td>'
    table_html += '</tr>'

table_html += '</table>'

# 显示表格
display(HTML(table_html))
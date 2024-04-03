import dgl
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
import dgllife.utils
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from dgllife.model import AttentiveFPGNN, AttentiveFPReadout, AttentiveFPPredictor, GATPredictor, GCNPredictor, \
    MPNNPredictor, NFPredictor, PAGTNPredictor, WeavePredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
# 加载数据
data = pd.read_csv('multi-pre-update.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 从数据中提取SMILES字符串和标签
smiles = data['SMILES']
labels = data.iloc[:, 1:]

# 先分割出训练集和一个临时集合，再将临时集合分为验证集和测试集
train_data, temp_data, train_label, temp_label = train_test_split(smiles, labels, test_size=0.2, random_state=0)
valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5, random_state=0)

# 重置索引
train_data = train_data.reset_index(drop=True)
train_label = train_label.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
valid_label = valid_label.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
test_label = test_label.reset_index(drop=True)

# label to tensor
train_label = [torch.tensor(train_label[col].values, dtype=torch.float32) for col in train_label.columns]
train_label = torch.stack(train_label, dim=1)

valid_label = [torch.tensor(valid_label[col].values, dtype=torch.float32) for col in valid_label.columns]
valid_label = torch.stack(valid_label, dim=1)

test_label = [torch.tensor(test_label[col].values, dtype=torch.float32) for col in test_label.columns]
test_label = torch.stack(test_label, dim=1)

def add_self_loops(graphs):
    graphs_with_loops = []
    for g in graphs:
        g_with_loops = dgl.add_self_loop(g)
        graphs_with_loops.append(g_with_loops)
    return graphs_with_loops

# Filter out invalid molecules before converting to graphs
def filter_valid_molecules(smiles_list, labels_list):
    valid_mols = []
    valid_labels = []
    for smiles, label in zip(smiles_list, labels_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Check if the molecule is valid
            valid_mols.append(mol)
            valid_labels.append(label)
        else:
            print(f"Invalid SMILES discarded: {smiles}")  # Optional: print or log the invalid SMILES
    return valid_mols, valid_labels

# Apply the filtering function to your datasets
train_mol, train_label = filter_valid_molecules(train_data, train_label)
valid_mol, valid_label = filter_valid_molecules(valid_data, valid_label)
test_mol, test_label = filter_valid_molecules(test_data, test_label)

# 设置原子和键的特征提取器
atom_featurizer = dgllife.utils.CanonicalAtomFeaturizer()
bond_featurizer = dgllife.utils.CanonicalBondFeaturizer()

# 将分子对象转换为图
train_g = [dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in train_mol]
valid_g = [dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in valid_mol]
test_g = [dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in test_mol]

# 创建数据集
train_dataset = list(zip(train_g, train_label))
valid_dataset = list(zip(valid_g, valid_label))
test_dataset = list(zip(test_g, test_label))

# 创建数据加载器
train_loader = GraphDataLoader(train_dataset, batch_size=64)
valid_loader = GraphDataLoader(valid_dataset, batch_size=64)
test_loader = GraphDataLoader(test_dataset, batch_size=64)

# 为需要自环的模型转换分子对象为图，并添加自环
train_g_with_loops = [dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in train_mol]
valid_g_with_loops = [dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in valid_mol]
test_g_with_loops = [dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in test_mol]

# 添加自环
train_g_with_loops = add_self_loops(train_g_with_loops)
valid_g_with_loops = add_self_loops(valid_g_with_loops)
test_g_with_loops = add_self_loops(test_g_with_loops)

# 为GAT, GCN, NF创建数据集（包含自环）
train_dataset_with_loops = list(zip(train_g_with_loops, train_label))
valid_dataset_with_loops = list(zip(valid_g_with_loops, valid_label))
test_dataset_with_loops = list(zip(test_g_with_loops, test_label))

# 为GAT, GCN, NF创建数据加载器（包含自环）
train_loader_with_loops = GraphDataLoader(train_dataset_with_loops, batch_size=64)
valid_loader_with_loops = GraphDataLoader(valid_dataset_with_loops, batch_size=64)
test_loader_with_loops = GraphDataLoader(test_dataset_with_loops, batch_size=64)

# 设置学习参数、模型、损失函数和优化器等
# 这部分代码保持不变，除非你需要调整模型或训练参数
# 继续设置学习参数

lr = 1e-3
num_epochs = 20

# 定义模型
Attentive_FP = AttentiveFPPredictor(node_feat_size=74, edge_feat_size=12, num_layers=2, num_timesteps=2, n_tasks=5)
GAT = GATPredictor(in_feats=74, classifier_hidden_feats=200, n_tasks=5, predictor_hidden_feats=200)
GCN = GCNPredictor(in_feats=74, classifier_hidden_feats=200, n_tasks=5, predictor_hidden_feats=200)
MPNN = MPNNPredictor(node_in_feats=74, node_out_feats=128, edge_hidden_feats=128, edge_in_feats=12, n_tasks=5)
NF = NFPredictor(in_feats=74, n_tasks=5, predictor_hidden_size=200)
PAGTN = PAGTNPredictor(node_in_feats=74, node_out_feats=128, node_hid_feats=128, edge_feats=12, depth=5, n_tasks=5)
Weave = WeavePredictor(node_in_feats=74, edge_in_feats=12, gnn_hidden_feats=64, graph_feats=64, num_gnn_layers=3, n_tasks=5)

# 移动模型到设备上
Attentive_FP.to(device)
GAT.to(device)
GCN.to(device)
MPNN.to(device)
NF.to(device)
PAGTN.to(device)
Weave.to(device)

from sklearn.metrics import roc_auc_score
loss_fn = nn.BCEWithLogitsLoss()


import csv
import os

def train_and_evaluate(model, device, train_loader, valid_loader, test_loader, optimizer, scheduler, loss_fn,
                       num_epochs=30, use_edge_features=True):
    best_valid_auc = 0
    model_name = type(model).__name__
    log_file = f'{model_name}_multi_trainlog.csv'

    # 创建CSV文件并写入标题行
    if not os.path.isfile(log_file):
        with open(log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Train ROC AUC', 'Valid Loss', 'Valid ROC AUC'])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_labels = []
        train_preds = []
        for bg, labels in train_loader:
            bg = bg.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if use_edge_features:
                n_feats = bg.ndata.pop('h').to(device)
                e_feats = bg.edata.pop('e').to(device)
                logits = model(bg, n_feats, e_feats)
            else:
                n_feats = bg.ndata.pop('h').to(device)
                logits = model(bg, n_feats)
            loss = loss_fn(logits.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            train_preds.extend(logits.squeeze(1).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 计算训练集平均损失和ROC AUC
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels, train_preds)

        # 验证集性能评估
        model.eval()
        total_valid_loss = 0
        valid_labels = []
        valid_preds = []
        with torch.no_grad():
            for bg, labels in valid_loader:
                bg = bg.to(device)
                labels = labels.to(device)
                if use_edge_features:
                    n_feats = bg.ndata.pop('h').to(device)
                    e_feats = bg.edata.pop('e').to(device)
                    logits = model(bg, n_feats, e_feats)
                else:
                    n_feats = bg.ndata.pop('h').to(device)
                    logits = model(bg, n_feats)
                valid_loss = loss_fn(logits.squeeze(1), labels.float())
                total_valid_loss += valid_loss.item() * labels.size(0)
                valid_preds.extend(logits.squeeze(1).cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

            avg_valid_loss = total_valid_loss / len(valid_loader.dataset)
            valid_auc = roc_auc_score(valid_labels, valid_preds)

            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                torch.save(model.state_dict(), f'multi_best_model_{model_name}.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train ROC AUC: {train_auc:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid ROC AUC: {valid_auc:.4f}')

        # 将指标写入CSV文件
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, train_auc, avg_valid_loss, valid_auc])

    # 加载最佳模型进行测试集评估
    model.load_state_dict(torch.load(f'multi_best_model_{type(model).__name__}.pth'))
    model.eval()
    test_labels = []
    test_preds = []
    with torch.no_grad():
        for bg, labels in test_loader:
            bg = bg.to(device)
            labels = labels.to(device)
            if use_edge_features:
                n_feats = bg.ndata.pop('h').to(device)
                e_feats = bg.edata.pop('e').to(device)
                labels = labels.to(torch.float32)
                logits = model(bg, n_feats, e_feats)
            else:
                n_feats = bg.ndata.pop('h').to(device)
                labels = labels.to(torch.float32)
                logits = model(bg, n_feats)
            test_preds.extend(logits.squeeze(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        test_auc = roc_auc_score(test_labels, test_preds)
        print(f'Test AUC for {type(model).__name__}: {test_auc:.4f}')

        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()  # 清理未使用的显存
        gc.collect()  # 强制进行垃圾收集

# 模型配置
models_to_train = {
    'Attentive_FP': (Attentive_FP, True),
    'GAT': (GAT, False),
    'GCN': (GCN, False),
    'MPNN': (MPNN, True),
    'NF': (NF, False),
    'PAGTN': (PAGTN, True),
    'Weave': (Weave, True),
}

for model_name, (model_instance, use_edge_features) in models_to_train.items():
    print(f"Training {model_name}")
    model_instance = model_instance.to(device)
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # 根据模型决定是否需要使用边特征以及是否需要自环的数据集
    if model_name in ['GAT', 'GCN', 'NF']:  # 这些模型在训练时不使用边特征，并需要自环
        train_loader_selected = train_loader_with_loops
        valid_loader_selected = valid_loader_with_loops
        test_loader_selected = test_loader_with_loops
    else:  # 其他模型在训练时使用边特征，不需要自环
        train_loader_selected = train_loader
        valid_loader_selected = valid_loader
        test_loader_selected = test_loader

    # 训练和评估模型
    print(f"Start training {model_name}...")
    train_and_evaluate(model_instance, device, train_loader_selected, valid_loader_selected, test_loader_selected,
                       optimizer, scheduler, loss_fn, num_epochs=30, use_edge_features=use_edge_features)
    # 清理显存和释放资源
    torch.cuda.empty_cache()
    gc.collect()
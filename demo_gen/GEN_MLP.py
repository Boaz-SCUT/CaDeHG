# 基因数据处理与分析教程 - 使用SurvPath项目现有模型

# 导入必要的库
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 为了导入项目中的模型，需要将项目根目录添加到Python路径
sys.path.append('../')  # 调整为你的SurvPath项目根目录

# 导入SurvPath项目中的模型
from models.model_MaskedOmics import MaskedOmics
from models.model_MLPOmics import MLPOmics
from models.model_SNNOmics import SNNOmics

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. 数据加载
print("1. 加载数据...")
metadata_path = "../datasets_csv/metadata/tcga_blca.csv"
data_path = "../datasets_csv/raw_rna_data/combine/blca/rna_clean.csv"

# 加载元数据
metadata = pd.read_csv(metadata_path)
print(f"元数据形状: {metadata.shape}")

# 加载基因表达数据
rna_data = pd.read_csv(data_path, index_col=0)
print(f"RNA数据形状: {rna_data.shape}")

# 2. 数据探索
print("\n2. 数据探索...")
# 检查生存时间和审查状态
if 'survival_months_dss' in metadata.columns and 'censorship_dss' in metadata.columns:
    plt.figure(figsize=(10, 6))
    # 区分审查和非审查样本
    censored = metadata[metadata['censorship_dss'] == 1]
    uncensored = metadata[metadata['censorship_dss'] == 0]
    
    plt.hist(uncensored['survival_months_dss'], alpha=0.5, bins=20, label='事件发生')
    plt.hist(censored['survival_months_dss'], alpha=0.5, bins=20, label='审查')
    plt.xlabel('生存时间(月)')
    plt.ylabel('样本数')
    plt.title('BLCA患者生存时间分布')
    plt.legend()
    plt.show()
    
    # 统计信息
    print(f"总样本数: {len(metadata)}")
    print(f"审查样本数: {len(censored)} ({len(censored)/len(metadata)*100:.2f}%)")
    print(f"非审查样本数: {len(uncensored)} ({len(uncensored)/len(metadata)*100:.2f}%)")
    print(f"平均生存时间: {metadata['survival_months_dss'].mean():.2f}月")
    print(f"中位生存时间: {metadata['survival_months_dss'].median():.2f}月")

# 3. 数据预处理
print("\n3. 数据预处理...")
# 标准化TCGA样本ID
def standardize_tcga_id(sample_id):
    if isinstance(sample_id, str) and sample_id.startswith('TCGA-'):
        return '-'.join(sample_id.split('-')[:3])
    return sample_id

# 应用到RNA数据索引
rna_data.index = rna_data.index.map(standardize_tcga_id)

# 应用到元数据
metadata['case_id'] = metadata['case_id'].map(standardize_tcga_id)

# 找出共同样本
common_samples = set(rna_data.index).intersection(set(metadata['case_id']))
print(f"共同样本数: {len(common_samples)}")

# 过滤数据，只保留共同样本
rna_data_filtered = rna_data.loc[list(common_samples)]
metadata_filtered = metadata[metadata['case_id'].isin(common_samples)].copy()

print(f"过滤后RNA数据形状: {rna_data_filtered.shape}")
print(f"过滤后元数据形状: {metadata_filtered.shape}")

# 4. 数据标准化
print("\n4. 数据标准化...")
scaler = MinMaxScaler(feature_range=(-1, 1))
rna_data_scaled = pd.DataFrame(
    scaler.fit_transform(rna_data_filtered),
    index=rna_data_filtered.index,
    columns=rna_data_filtered.columns
)

print(f"标准化后数据的最小值: {rna_data_scaled.values.min()}")
print(f"标准化后数据的最大值: {rna_data_scaled.values.max()}")

# 5. 加载路径组成信息
print("\n5. 加载路径组成信息...")
try:
    pathway_comp_path = "../datasets_csv/pathway_compositions/combine_comps.csv"
    pathway_comp = pd.read_csv(pathway_comp_path, index_col=0)
    print(f"路径组成数据形状: {pathway_comp.shape}")
    
    # 检查共同基因
    common_genes = set(rna_data_scaled.columns).intersection(set(pathway_comp.index))
    print(f"共同基因数: {len(common_genes)}")
    
    # 过滤路径组成，只保留共同基因
    pathway_comp_filtered = pathway_comp.loc[list(common_genes)]
    print(f"过滤后路径组成形状: {pathway_comp_filtered.shape}")
    
    composition_df = pathway_comp_filtered  # 用于MaskedOmics模型
    
except FileNotFoundError:
    print(f"未找到路径组成文件: {pathway_comp_path}")
    pathway_comp_filtered = None
    composition_df = None

# 6. 准备数据集类
class GeneExpressionDataset(Dataset):
    def __init__(self, gene_expr, metadata, label_col='survival_months_dss', 
                censorship_col='censorship_dss', n_bins=4):
        # 确保元数据中有case_id列
        if 'case_id' not in metadata.columns:
            raise ValueError("metadata必须包含'case_id'列")
        
        # 获取共同样本
        common_samples = list(set(gene_expr.index) & set(metadata['case_id']))
        self.gene_expr = gene_expr.loc[common_samples]
        
        # 创建case_id到metadata行索引的映射
        self.metadata = metadata[metadata['case_id'].isin(common_samples)].copy()
        self.metadata.set_index('case_id', inplace=True)
        
        # 存储列名
        self.label_col = label_col
        self.censorship_col = censorship_col
        
        # 创建分位数标签
        self._create_discrete_labels(n_bins)
        
        # 样本ID
        self.sample_ids = common_samples
        
    def _create_discrete_labels(self, n_bins):
        """创建生存时间的离散标签"""
        times = self.metadata[self.label_col]
        uncensored = self.metadata[self.metadata[self.censorship_col] < 1]
        
        # 使用未审查样本计算分位数
        q_bins = np.percentile(uncensored[self.label_col], np.linspace(0, 100, n_bins+1))
        q_bins[0] = times.min() - 0.001  # 确保最小值能被包含
        q_bins[-1] = times.max() + 0.001  # 确保最大值能被包含
        
        # 创建离散标签
        self.metadata['disc_label'] = pd.cut(times, bins=q_bins, labels=False, include_lowest=True)
        self.metadata['disc_label'] = self.metadata['disc_label'].fillna(0).astype(int)
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 获取基因表达数据
        gene_expr = torch.tensor(self.gene_expr.loc[sample_id].values, dtype=torch.float32)
        
        # 获取生存时间和审查状态
        survival_time = float(self.metadata.loc[sample_id, self.label_col])
        censor = float(self.metadata.loc[sample_id, self.censorship_col])
        
        # 获取离散标签
        label = int(self.metadata.loc[sample_id, 'disc_label'])
        
        return (
            torch.zeros((1, 1)),  # 占位符，保持与项目API一致
            gene_expr,  # 基因表达数据
            torch.tensor([label], dtype=torch.long),  # 离散标签
            torch.tensor([survival_time], dtype=torch.float32),  # 生存时间
            torch.tensor([censor], dtype=torch.float32),  # 审查状态
            ("N/A", "N/A", "N/A")  # 占位符，保持与项目API一致
        )

# 7. 准备数据集
print("\n7. 准备数据集...")
# 分割样本ID
sample_ids = rna_data_scaled.index.tolist()
train_ids, val_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)

# 提取训练和验证数据
train_expr = rna_data_scaled.loc[train_ids]
val_expr = rna_data_scaled.loc[val_ids]

# 创建数据集
train_dataset = GeneExpressionDataset(train_expr, metadata_filtered)
val_dataset = GeneExpressionDataset(val_expr, metadata_filtered)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 8. 定义生存损失函数
class NLLSurvLoss(nn.Module):
    def __init__(self, alpha=0.0, eps=1e-7):
        super(NLLSurvLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, h, y, t, c):
        # h: 模型输出的logits
        # y: 离散的生存时间指标
        # t: 连续的生存时间
        # c: 审查状态
        
        # 将 h 转换为 hazards
        hazards = torch.sigmoid(h)
        
        # 计算生存率 S(t) = exp(-H(t))
        S = torch.cumprod(1 - hazards, dim=1)
        
        # 添加 S(-1) = 1
        S_padded = torch.cat([torch.ones_like(y, dtype=torch.float32), S], 1)
        
        # 获取特定时间点的值
        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=self.eps)
        
        # 计算未审查损失和审查损失
        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = -c * torch.log(s_this)
        
        # 合并损失
        neg_l = censored_loss + uncensored_loss
        
        # 应用 alpha 加权
        loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss
        
        return loss.sum()

# 9. 训练和评估函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for data in train_loader:
            # 解包数据
            _, x_gene, y, survival_time, censor, _ = data
            x_gene = x_gene.to(device)
            y = y.to(device)
            survival_time = survival_time.to(device)
            censor = censor.to(device)
            
            # 调用模型前向传播
            outputs = model(data_omics=x_gene)
            
            # 计算损失
            loss = criterion(outputs, y, survival_time, censor)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                # 解包数据
                _, x_gene, y, survival_time, censor, _ = data
                x_gene = x_gene.to(device)
                y = y.to(device)
                survival_time = survival_time.to(device)
                censor = censor.to(device)
                
                # 调用模型前向传播
                outputs = model(data_omics=x_gene)
                
                # 计算损失
                loss = criterion(outputs, y, survival_time, censor)
                
                running_loss += loss.item()
        
        epoch_val_loss = running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return model, train_losses, val_losses

def compute_c_index(model, data_loader, device):
    model.eval()
    all_risk_scores = []
    all_survival_times = []
    all_censors = []
    
    with torch.no_grad():
        for data in data_loader:
            # 解包数据
            _, x_gene, _, survival_time, censor, _ = data
            x_gene = x_gene.to(device)
            
            # 调用模型前向传播
            outputs = model(data_omics=x_gene)
            
            # 计算风险分数(负的累积生存率)
            hazards = torch.sigmoid(outputs)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1)
            
            all_risk_scores.append(risk.cpu().numpy())
            all_survival_times.append(survival_time.numpy())
            all_censors.append(censor.numpy())
    
    # 合并所有批次的结果
    all_risk_scores = np.concatenate(all_risk_scores)
    all_survival_times = np.concatenate(all_survival_times)
    all_censors = np.concatenate(all_censors)
    
    # 计算一致性指数
    from lifelines.utils import concordance_index
    c_index = concordance_index(all_survival_times, all_risk_scores, 1 - all_censors)
    
    return c_index

# 10. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 11. 定义模型参数
input_dim = rna_data_scaled.shape[1]  # 基因数量
n_classes = 4  # 生存时间分箱数

# 12. 训练MLPOmics模型
print("\n12. 训练MLPOmics模型...")
mlp_model = MLPOmics(
    input_dim=input_dim,
    n_classes=n_classes,
    projection_dim=256,
    dropout=0.3
)

criterion = NLLSurvLoss(alpha=0.5)
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)

mlp_model, mlp_train_losses, mlp_val_losses = train_model(
    mlp_model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device
)

mlp_c_index = compute_c_index(mlp_model, val_loader, device)
print(f"MLPOmics模型验证集C-index: {mlp_c_index:.4f}")

# 13. 训练SNNOmics模型
print("\n13. 训练SNNOmics模型...")
snn_model = SNNOmics(
    omic_input_dim=input_dim,
    model_size_omic='small',
    n_classes=n_classes
)

optimizer = optim.Adam(snn_model.parameters(), lr=0.001, weight_decay=1e-5)

snn_model, snn_train_losses, snn_val_losses = train_model(
    snn_model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device
)

snn_c_index = compute_c_index(snn_model, val_loader, device)
print(f"SNNOmics模型验证集C-index: {snn_c_index:.4f}")

# 14. 如果有路径组成数据，训练MaskedOmics模型
if composition_df is not None:
    print("\n14. 训练MaskedOmics模型...")
    masked_model = MaskedOmics(
        device=device,
        df_comp=composition_df,
        input_dim=len(common_genes),
        dim_per_path_1=16,
        dim_per_path_2=32,
        dropout=0.3,
        num_classes=n_classes
    )

    optimizer = optim.Adam(masked_model.parameters(), lr=0.001, weight_decay=1e-5)

    # 需要使用共同基因的子集
    # 创建一个新的数据集类，只包含共同基因
    class CommonGeneDataset(Dataset):
        def __init__(self, dataset, common_genes):
            self.dataset = dataset
            self.common_genes = common_genes
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            placeholder, gene_expr, label, survival_time, censor, clinical = self.dataset[idx]
            
            # 只选择共同基因
            gene_indices = [i for i, gene in enumerate(self.dataset.gene_expr.columns) if gene in self.common_genes]
            common_gene_expr = gene_expr[gene_indices]
            
            return placeholder, common_gene_expr, label, survival_time, censor, clinical
    
    # 创建只包含共同基因的数据集
    common_gene_train_dataset = CommonGeneDataset(train_dataset, common_genes)
    common_gene_val_dataset = CommonGeneDataset(val_dataset, common_genes)
    
    # 创建数据加载器
    common_gene_train_loader = DataLoader(common_gene_train_dataset, batch_size=16, shuffle=True)
    common_gene_val_loader = DataLoader(common_gene_val_dataset, batch_size=16, shuffle=False)
    
    masked_model, masked_train_losses, masked_val_losses = train_model(
        masked_model, common_gene_train_loader, common_gene_val_loader, 
        criterion, optimizer, num_epochs=20, device=device
    )
    
    masked_c_index = compute_c_index(masked_model, common_gene_val_loader, device)
    print(f"MaskedOmics模型验证集C-index: {masked_c_index:.4f}")
    
    # 绘制所有模型的损失曲线
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_train_losses, label='MLPOmics')
    plt.plot(snn_train_losses, label='SNNOmics')
    plt.plot(masked_train_losses, label='MaskedOmics')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_val_losses, label='MLPOmics')
    plt.plot(snn_val_losses, label='SNNOmics')
    plt.plot(masked_val_losses, label='MaskedOmics')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('验证损失')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"MLPOmics模型C-index: {mlp_c_index:.4f}")
    print(f"SNNOmics模型C-index: {snn_c_index:.4f}")
    print(f"MaskedOmics模型C-index: {masked_c_index:.4f}")
else:
    # 只比较MLPOmics和SNNOmics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_train_losses, label='MLPOmics')
    plt.plot(snn_train_losses, label='SNNOmics')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_val_losses, label='MLPOmics')
    plt.plot(snn_val_losses, label='SNNOmics')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('验证损失')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"MLPOmics模型C-index: {mlp_c_index:.4f}")
    print(f"SNNOmics模型C-index: {snn_c_index:.4f}")

# 15. 保存模型和结果
print("\n15. 保存模型和结果...")
os.makedirs('models', exist_ok=True)

# 保存MLPOmics模型
torch.save(mlp_model.state_dict(), 'models/mlp_omics_model.pt')

# 保存SNNOmics模型
torch.save(snn_model.state_dict(), 'models/snn_omics_model.pt')

# 如果有MaskedOmics模型，也保存它
if 'masked_model' in locals():
    torch.save(masked_model.state_dict(), 'models/masked_omics_model.pt')

print("\n==== 基因数据处理与分析总结 ====")
print(f"数据集: BLCA (膀胱癌)")
print(f"样本数: {len(metadata_filtered)}")
print(f"基因数: {rna_data_scaled.shape[1]}")
if pathway_comp_filtered is not None:
    print(f"路径数: {pathway_comp_filtered.shape[1]}")
print("\n模型性能:")
print(f"MLPOmics模型C-index: {mlp_c_index:.4f}")
print(f"SNNOmics模型C-index: {snn_c_index:.4f}")
if 'masked_c_index' in locals():
    print(f"MaskedOmics模型C-index: {masked_c_index:.4f}")

print("\n下一步可能的改进:")
print("1. 尝试不同的超参数组合")
print("2. 使用交叉验证获得更稳健的性能评估")
print("3. 整合临床特征以提高预测性能")
print("4. 尝试更多的生存分析评估指标")
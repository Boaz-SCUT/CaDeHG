import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from einops import reduce

from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class DoubleMediatorSurvPath(nn.Module):
    def __init__(
        self, 
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=768,
        dropout=0.2,
        num_classes=4,
        pathway_dim=256,
        interaction_dim=256,
        n_dictionary_items=5,
        intervention_alpha=0,
        use_wsi=False,
        bio_constraints=True,
        omic_names = [],
        ):
        super(DoubleMediatorSurvPath, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_wsi = use_wsi
        self.intervention_alpha = intervention_alpha
        self.bio_constraints = bio_constraints
        self.pathway_dim = pathway_dim
        self.interaction_dim = interaction_dim
        
        #---> 通路名称处理，用于后续解释
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        # 初始化每个通路的编码器
        self.init_per_path_model(omic_sizes)
        
        # 改进：添加特征扩展层，增加表达能力
        self.feature_expansion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pathway_dim, self.pathway_dim*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.pathway_dim*2, self.pathway_dim)
            ) for _ in range(self.num_pathways)
        ])
        
        # 交叉注意力层：自身与自身的交叉注意力
        self.cross_attender = nn.MultiheadAttention(
            embed_dim=self.pathway_dim,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
        # 自注意力层
        self.self_attender = nn.MultiheadAttention(
            embed_dim=self.pathway_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 改进：通路图层，使用更复杂的边特征计算
        self.pathway_graph_layer = nn.Sequential(
            nn.Linear(2 * self.pathway_dim, self.interaction_dim),
            nn.LayerNorm(self.interaction_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.interaction_dim, self.interaction_dim//2),
            nn.ReLU(),
            nn.Linear(self.interaction_dim//2, 1)
        )
        
        # 改进：添加残差连接
        # self.residual_projector = nn.Linear(self.num_pathways * omic_sizes[0], self.pathway_dim * self.num_pathways)
        # 1x9745 -> 1x256
        self.residual_projector = nn.Linear(9745, self.pathway_dim * self.num_pathways)
        
        # 改进：分层分类器
        self.pre_classifier = nn.Sequential(
            nn.Linear(self.pathway_dim, self.pathway_dim // 2),
            nn.LayerNorm(self.pathway_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.pathway_dim, self.pathway_dim // 2),
            nn.LayerNorm(self.pathway_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.pathway_dim // 2, self.num_classes)
        )
        
        # 改进：添加通路重要性学习
        self.pathway_importance = nn.Parameter(torch.ones(self.num_pathways) / self.num_pathways)
            
    def init_per_path_model(self, omic_sizes):
        """初始化每个通路的编码器"""
        hidden = [256, self.pathway_dim]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
    
    def forward(self, **kwargs):
        """前向传播函数"""
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways+1)] # 获取每个通路的输入,331
        
        # 保存原始输入用于残差连接
        original_input = torch.cat([x.float() for x in x_omic], dim=1) if all(x.dim() > 1 for x in x_omic) else torch.cat([x.float().unsqueeze(0) for x in x_omic], dim=1)
        
        return_attn = kwargs.get("return_attn", False)
        
        # 第一层中介: 通路活性计算并应用通路重要性权重
        h_omic = []
        for idx, sig_feat in enumerate(x_omic):
            # 计算通路嵌入
            pathway_embed = self.sig_networks[idx](sig_feat.float())
            # 应用通路重要性权重
            weighted_embed = pathway_embed * F.softmax(self.pathway_importance, dim=0)[idx]
            # 特征扩展 - 捕获更复杂的通路内关系
            expanded_embed = self.feature_expansion[idx](weighted_embed) + weighted_embed  # 残差连接
            h_omic.append(expanded_embed)
        
        # 堆叠所有通路嵌入
        h_omic_bag = torch.stack(h_omic, dim=0)  # [num_pathways, batch_size, pathway_dim] 
        
        # 处理批次维度
        if h_omic_bag.dim() == 2:  # 如果只有一个样本
            h_omic_bag = h_omic_bag.unsqueeze(1)  # [num_pathways, 1, pathway_dim]
        
        # 转置为 [batch_size, num_pathways, pathway_dim]
        h_omic_bag = h_omic_bag.permute(1, 0, 2)
        
        # 应用自注意力处理通路间关系 - 使用masked注意力确保通路只关注其他相关通路
        attn_mask = ~torch.eye(self.num_pathways, dtype=torch.bool, device=h_omic_bag.device)
        h_omic_flat = h_omic_bag.reshape(-1, self.num_pathways, self.pathway_dim)  # 确保形状正确
        
        # 改进：应用交叉注意力而不是自注意力 - 通路互相传递信息
        # h_omic_cross, cross_attn_weights = self.cross_attender(
        #     query=h_omic_flat,
        #     key=h_omic_flat,
        #     value=h_omic_flat
        # )
        
        # 应用自注意力进一步精炼特征
        h_omic_attn, self_attn_weights = self.self_attender(
            query=h_omic_flat,
            key=h_omic_flat,
            value=h_omic_flat
        ) # h_omic_attn:[batch_size, num_pathways, pathway_dim] self_attn_weights:[batch_size, num_pathways, num_pathways]
        # self_attn_weights加权累加所有通路特征
        h_weighted = torch.bmm(self_attn_weights, h_omic_attn)  # h_weighted: [batch_size, num_pathways, pathway_dim]
        h_omic_attn = h_weighted.sum(dim=1) # h_omic_attn: [batch_size, pathway_dim]
        pathway_features = h_omic_attn # [batch_size, num_pathways, pathway_dim]
        # 分类
        logits = self.classifier(pathway_features)
        
        # # 构建通路图
        # batch_size = h_omic_attn.size(0)
        # adj_matrices = []
        # pathway_features_list = []
        
        # for b in range(batch_size):
        #     # 构建每个样本的通路图
        #     adj_matrix = self._build_pathway_graph(h_omic_attn[b])
        #     norm_adj = self._normalize_adj(adj_matrix)
            
        #     # 图传播 - 通路特征通过图结构进行传递
        #     pathway_features = torch.matmul(norm_adj, h_omic_attn[b])  # [num_pathways, pathway_dim]
            
        #     adj_matrices.append(adj_matrix)
        #     pathway_features_list.append(pathway_features)
            
        # # 堆叠所有批次的结果
        # pathway_features = torch.stack(pathway_features_list, dim=0)  # [batch_size, num_pathways, pathway_dim]
        
        # # 特征展平
        # pathway_features = pathway_features.reshape(batch_size, -1)  # [batch_size, num_pathways * pathway_dim]
        
        # # 残差连接 - 允许原始基因信息直接影响预测
        # if original_input.size(1) != self.pathway_dim * self.num_pathways:# original_input [1,9745],
        #     residual = self.residual_projector(original_input)
        # else:
        #     residual = original_input
        
        # 分层分类，先进行特征转换
        # pre_logits = self.pre_classifier(pathway_features)
        
        # 最终预测时添加残差连接
        # logits = self.classifier(pre_logits + 0.1 * residual.reshape(batch_size, -1)[:, :pre_logits.size(1)])
        # logits = self.classifier(pre_logits)
        
        return logits
    
    def _build_pathway_graph(self, pathway_features):
        """构建通路关系图 - 高效批处理版本"""
        num_pathways = pathway_features.shape[0]
        feature_dim = pathway_features.shape[1]
        
        # 使用广播机制创建所有通路对的组合特征
        features_i = pathway_features.unsqueeze(1)  # [num_pathways, 1, feature_dim]
        features_j = pathway_features.unsqueeze(0)  # [1, num_pathways, feature_dim]
        
        # 合并特征 [num_pathways, num_pathways, 2*feature_dim]
        combined_features = torch.cat([
            features_i.expand(num_pathways, num_pathways, feature_dim),
            features_j.expand(num_pathways, num_pathways, feature_dim)
        ], dim=2)
        
        # 重塑为二维张量以便批处理 [num_pathways*num_pathways, 2*feature_dim]
        combined_features_flat = combined_features.reshape(-1, 2*feature_dim)
        
        # 一次性计算所有关系分数
        scores = self.pathway_graph_layer(combined_features_flat)
        
        # 重塑回原始形状
        scores = scores.reshape(num_pathways, num_pathways)
        
        # 创建对角线掩码（1表示对角线位置，0表示其他位置）
        diag_mask = torch.eye(num_pathways, device=scores.device)
        
        # 对角线位置为1.0，其他位置应用sigmoid
        adj_matrix = diag_mask + (1 - diag_mask) * torch.sigmoid(scores)
        
        return adj_matrix
    
    def _normalize_adj(self, adj):
        """归一化邻接矩阵，用于GCN"""
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # 计算度矩阵
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum + 1e-6, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # D^(-1/2) * A * D^(-1/2)
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    def compute_bio_constraints_loss(self, adj_matrices, prior_knowledge=None):
        """计算生物学约束损失"""
        if not self.bio_constraints:
            return torch.tensor(0.0, device=adj_matrices[0].device)
        
        # 计算平均损失
        total_loss = 0.0
        for adj_matrix in adj_matrices:
            # 稀疏性约束: 鼓励邻接矩阵稀疏
            sparsity_loss = torch.mean(torch.abs(adj_matrix))
            
            # 如果有先验知识，添加结构约束
            structure_loss = torch.tensor(0.0, device=adj_matrix.device)
            if prior_knowledge is not None:
                structure_loss = F.mse_loss(adj_matrix, prior_knowledge)
            
            # 总损失
            total_loss += structure_loss + 0.01 * sparsity_loss
            
        return total_loss / len(adj_matrices)
    
    def captum(self, *args):
        """与原始SurvPath兼容的Captum接口"""
        # 如果使用WSI，需要单独处理
        if self.use_wsi:
            wsi = args[-1]
            omics_args = args[:-1]
        else:
            wsi = None
            omics_args = args
        
        # 构建输入
        input_dict = {}
        for i, omic in enumerate(omics_args):
            if i < self.num_pathways:  # 仅处理有效的通路输入
                input_dict[f'x_omic{i+1}'] = omic
        
        if self.use_wsi:
            input_dict['x_path'] = wsi
        
        input_dict['return_attn'] = False
        
        # 前向传播
        logits = self.forward(**input_dict)
        
        # 计算风险
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        
        return risk
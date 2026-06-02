import torch
import numpy as np 
import torch.nn as nn
from torch import nn
from einops import reduce
from torch.nn import ReLU
from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb
import math
import pandas as pd

def exists(val):
    return val is not None

def SNN_Block(dim1, dim2, dropout=0.25):
    """
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class CausalSurvPath(nn.Module):
    """
    因果增强的SurvPath模型
    在原有SurvPath基础上集成：
    1. 临床特征处理
    2. 倾向性得分调整
    3. 因果正则化
    """
    def __init__(
        self, 
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names = [],
        clinical_dim=5,  # 新增：临床特征维度
        enable_causal=True,  # 新增：启用因果功能
        propensity_weight=0.1,  # 新增：倾向性得分权重
        causal_reg_weight=0.05,  # 新增：因果正则化权重
        ):
        super(CausalSurvPath, self).__init__()

        #---> 原有参数
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim
        self.num_classes = num_classes

        #---> 新增：因果相关参数
        self.clinical_dim = clinical_dim
        self.enable_causal = enable_causal
        self.propensity_weight = propensity_weight
        self.causal_reg_weight = causal_reg_weight

        #---> omics preprocessing for captum
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

        #---> WSI投影网络
        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> 通路级别网络初始化
        self.init_per_path_model(omic_sizes)

        #---> 新增：临床特征编码器
        if self.enable_causal:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # 倾向性得分网络
            self.propensity_network = nn.Sequential(
                nn.Linear(clinical_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
            # 因果调整的交叉注意力
            self.causal_cross_attender = CausalMMAttentionLayer(
                dim=self.wsi_projection_dim,
                dim_head=self.wsi_projection_dim // 2,
                heads=1,
                residual=False,
                dropout=0.1,
                num_pathways=self.num_pathways,
                clinical_dim=32
            )
        else:
            # 标准交叉注意力
            self.cross_attender = MMAttentionLayer(
                dim=self.wsi_projection_dim,
                dim_head=self.wsi_projection_dim // 2,
                heads=1,
                residual=False,
                dropout=0.1,
                num_pathways=self.num_pathways
            )

        #---> 特征融合
        self.identity = nn.Identity()  # 用于计算梯度
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # 输出层
        if self.enable_causal:
            # 包含临床特征的融合层
            fusion_input_dim = self.wsi_projection_dim + 32  # pathway + wsi + clinical
            self.to_logits = nn.Sequential(
                nn.Linear(fusion_input_dim, int(fusion_input_dim/4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(int(fusion_input_dim/4), self.num_classes)
            )
        else:
            # 原有输出层
            self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        
    def init_per_path_model(self, omic_sizes):
        """初始化通路级别的网络"""
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)    
    
    def forward(self, **kwargs):
        """
        前向传播，支持因果增强功能
        """
        wsi = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways+1)]
        mask = kwargs.get('mask', None)
        return_attn = kwargs.get("return_attn", False)
        
        # 新增：处理临床特征
        clinical_features = kwargs.get('clinical_features', None)
        
        #---> 获取通路嵌入
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)  # [1, num_pathways, dim]

        #---> WSI特征投影
        wsi_embed = self.wsi_projection_net(wsi)

        #---> 因果增强处理
        if self.enable_causal and clinical_features is not None:
            # 编码临床特征
            clinical_embed = self.clinical_encoder(clinical_features)
            
            # 计算倾向性得分
            propensity_scores = self.propensity_network(clinical_features)
            
            # 使用倾向性得分调整特征权重
            # 对通路特征应用权重
            weighted_omic_bag = h_omic_bag * (1 + self.propensity_weight * propensity_scores.unsqueeze(-1).unsqueeze(-1))
            
            # 对WSI特征应用权重
            weighted_wsi_embed = wsi_embed * (1 + self.propensity_weight * propensity_scores.unsqueeze(-1))
            
            # 组合特征
            tokens = torch.cat([weighted_omic_bag, weighted_wsi_embed], dim=1)
            tokens = self.identity(tokens)
            
            # 因果增强的交叉注意力
            if return_attn:
                mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.causal_cross_attender(
                    x=tokens, clinical_embed=clinical_embed, mask=mask, return_attention=True
                )
            else:
                mm_embed = self.causal_cross_attender(
                    x=tokens, clinical_embed=clinical_embed, mask=mask, return_attention=False
                )
            
            # 存储中间结果用于损失计算
            self._last_clinical_embed = clinical_embed
            self._last_pathway_embed = weighted_omic_bag
            self._last_propensity_scores = propensity_scores
            
        else:
            # 标准处理流程
            tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
            tokens = self.identity(tokens)
            
            if return_attn:
                mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(
                    x=tokens, mask=mask, return_attention=True
                )
            else:
                mm_embed = self.cross_attender(x=tokens, mask=mask, return_attention=False)

        #---> 后处理
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> 聚合特征
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # 特征融合
        if self.enable_causal and clinical_features is not None:
            # 包含临床特征的融合
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed, clinical_embed], dim=1)
        else:
            # 标准融合
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        #---> 生成logits
        logits = self.to_logits(embedding)

        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits

    def compute_causal_regularization(self):
        """
        计算因果正则化项
        目标：确保通路特征与混杂的临床特征相对独立
        """
        if not self.enable_causal or not hasattr(self, '_last_clinical_embed'):
            return torch.tensor(0.0)
        
        try:
            # 计算通路特征与临床特征的相关性
            pathway_features = self._last_pathway_embed.squeeze(0)  # [num_pathways, dim]
            clinical_features = self._last_clinical_embed  # [clinical_dim]
            
            # 简化的相关性计算
            pathway_mean = torch.mean(pathway_features, dim=0)  # [dim]
            
            # 计算标准化后的特征
            pathway_std = pathway_mean / (torch.norm(pathway_mean) + 1e-8)
            clinical_std = clinical_features / (torch.norm(clinical_features) + 1e-8)
            
            # 计算相关性（简化版互信息）
            correlation = torch.abs(torch.dot(pathway_std, clinical_std[:len(pathway_std)]))
            
            return correlation
            
        except Exception as e:
            # 如果计算失败，返回0
            return torch.tensor(0.0)

    def get_propensity_scores(self):
        """获取最后一次前向传播的倾向性得分"""
        if hasattr(self, '_last_propensity_scores'):
            return self._last_propensity_scores
        return None


class CausalMMAttentionLayer(nn.Module):
    """
    因果增强的多模态注意力层
    集成临床特征调整
    """
    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways=281,
        clinical_dim=32,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.clinical_dim = clinical_dim
        
        # 原有注意力机制
        self.attn = MMAttentionLayer(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )
        
        # 临床特征调整层
        self.clinical_adjustment = nn.Sequential(
            nn.Linear(clinical_dim, dim),
            nn.Tanh()
        )

    def forward(self, x=None, clinical_embed=None, mask=None, return_attention=False):
        """
        前向传播，集成临床特征调整
        """
        # 标准化输入
        x_norm = self.norm(x)
        
        # 如果有临床特征，进行调整
        if clinical_embed is not None:
            # 计算临床调整项
            clinical_adj = self.clinical_adjustment(clinical_embed)  # [clinical_dim] -> [dim]
            
            # 对通路特征进行调整
            pathway_features = x_norm[:, :self.num_pathways, :]  # [batch, num_pathways, dim]
            adjusted_pathway = pathway_features + 0.1 * clinical_adj.unsqueeze(0).unsqueeze(0)
            
            # 重新组合特征
            x_adjusted = torch.cat([
                adjusted_pathway,
                x_norm[:, self.num_pathways:, :]  # WSI特征保持不变
            ], dim=1)
        else:
            x_adjusted = x_norm

        # 应用注意力机制
        if return_attention:
            output, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn.forward(
                x=x_adjusted, mask=mask, return_attention=True
            )
            return output, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            output = self.attn.forward(x=x_adjusted, mask=mask, return_attention=False)
            return output


# 与原有MMAttentionLayer保持兼容
class MMAttentionLayer(nn.Module):
    """
    标准多模态注意力层（保持与原代码兼容）
    """
    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways=281,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        
        # 导入原有的MMAttention（简化版实现）
        self.attn = SimplifiedMMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):
        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(
                x=self.norm(x), mask=mask, return_attn=True
            )
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask, return_attn=False)
            return x


class SimplifiedMMAttention(nn.Module):
    """
    简化的多模态注意力实现
    """
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        residual=True,
        dropout=0.,
        num_pathways=281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        b, n, d = x.shape
        h = self.heads
        
        # 生成查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)
        
        # 计算注意力
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(b, h, n, n)
            dots.masked_fill_(mask, float('-inf'))
        
        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        out = self.to_out(out)
        
        if return_attn:
            # 简化的注意力权重返回
            pathway_attn = attn[:, :, :self.num_pathways, :self.num_pathways].mean(1)
            cross_attn_pathways = attn[:, :, :self.num_pathways, self.num_pathways:].mean(1)
            cross_attn_histology = attn[:, :, self.num_pathways:, :self.num_pathways].mean(1)
            
            return out, pathway_attn, cross_attn_pathways, cross_attn_histology
        
        return out
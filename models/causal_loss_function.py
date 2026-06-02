import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import combinations
import pdb

class CausalNLLSurvLoss(nn.Module):
    """
    因果增强的负对数似然生存损失函数
    在原有NLLSurvLoss基础上添加：
    1. 因果正则化项
    2. 倾向性得分平衡
    3. 环境不变性约束（为第二阶段IRM做准备）
    """
    def __init__(self, 
                 alpha=0.0, 
                 eps=1e-7, 
                 reduction='sum',
                 causal_reg_weight=0.05,
                 propensity_balance_weight=0.1,
                 enable_causal=True):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.causal_reg_weight = causal_reg_weight
        self.propensity_balance_weight = propensity_balance_weight
        self.enable_causal = enable_causal
        
        # 基础NLL损失
        self.base_nll_loss = NLLSurvLoss(alpha=alpha, eps=eps, reduction=reduction)

    def forward(self, h, y, t, c, model=None, clinical_features=None):
        """
        计算因果增强的生存损失
        
        Args:
            h: 模型预测的hazard logits [batch_size, n_classes]
            y: 离散化的生存时间标签 [batch_size, 1]
            t: 实际生存时间 [batch_size, 1]
            c: 删失指示符 [batch_size, 1]
            model: 模型实例（用于计算因果正则化）
            clinical_features: 临床特征 [batch_size, clinical_dim]
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 基础生存损失
        base_loss = self.base_nll_loss(h, y, t, c)
        
        loss_dict = {
            'base_loss': base_loss,
            'causal_reg_loss': torch.tensor(0.0, device=h.device),
            'propensity_balance_loss': torch.tensor(0.0, device=h.device),
            'total_loss': base_loss
        }
        
        if not self.enable_causal or model is None:
            return loss_dict
        
        # 因果正则化损失
        if hasattr(model, 'compute_causal_regularization'):
            causal_reg_loss = model.compute_causal_regularization()
            loss_dict['causal_reg_loss'] = causal_reg_loss
        
        # 倾向性得分平衡损失
        if clinical_features is not None:
            propensity_balance_loss = self.compute_propensity_balance_loss(model, clinical_features, y, c)
            loss_dict['propensity_balance_loss'] = propensity_balance_loss
        
        # 总损失
        total_loss = (base_loss + 
                     self.causal_reg_weight * loss_dict['causal_reg_loss'] +
                     self.propensity_balance_weight * loss_dict['propensity_balance_loss'])
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def compute_propensity_balance_loss(self, model, clinical_features, y, c):
        """
        计算倾向性得分平衡损失
        目标：确保不同结果组的倾向性得分分布相似
        """
        try:
            propensity_scores = model.get_propensity_scores()
            if propensity_scores is None:
                return torch.tensor(0.0, device=clinical_features.device)
            
            # 将患者分为不同的结果组
            # 未删失 vs 删失
            uncensored_mask = (c.squeeze() == 0)
            censored_mask = (c.squeeze() == 1)
            
            if uncensored_mask.sum() > 0 and censored_mask.sum() > 0:
                # 计算两组的倾向性得分分布差异
                uncensored_prop = propensity_scores[uncensored_mask]
                censored_prop = propensity_scores[censored_mask]
                
                # 使用KL散度衡量分布差异（简化版）
                mean_diff = torch.abs(uncensored_prop.mean() - censored_prop.mean())
                var_diff = torch.abs(uncensored_prop.var() - censored_prop.var())
                
                balance_loss = mean_diff + 0.5 * var_diff
                return balance_loss
            else:
                return torch.tensor(0.0, device=clinical_features.device)
                
        except Exception as e:
            # 如果计算失败，返回0
            return torch.tensor(0.0, device=clinical_features.device)


class NLLSurvLoss(nn.Module):
    """
    原有的负对数似然生存损失函数
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, h, y, t, c):
        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    标准的负对数似然生存损失计算
    """
    # 确保数据类型正确
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)
    
    # 添加padding确保索引不越界
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    
    # 计算相关概率
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    
    # 计算损失
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    
    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


class CausalSurvivalMetrics:
    """
    因果增强的生存分析评估指标
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置累积指标"""
        self.total_base_loss = 0.0
        self.total_causal_reg_loss = 0.0
        self.total_propensity_balance_loss = 0.0
        self.total_samples = 0
    
    def update(self, loss_dict, batch_size):
        """更新累积指标"""
        self.total_base_loss += loss_dict['base_loss'].item() * batch_size
        self.total_causal_reg_loss += loss_dict['causal_reg_loss'].item() * batch_size
        self.total_propensity_balance_loss += loss_dict['propensity_balance_loss'].item() * batch_size
        self.total_samples += batch_size
    
    def compute(self):
        """计算平均指标"""
        if self.total_samples == 0:
            return {}
        
        return {
            'avg_base_loss': self.total_base_loss / self.total_samples,
            'avg_causal_reg_loss': self.total_causal_reg_loss / self.total_samples,
            'avg_propensity_balance_loss': self.total_propensity_balance_loss / self.total_samples,
        }
    
    def compute_propensity_score_balance(self, propensity_scores, treatment_groups):
        """
        计算倾向性得分平衡性指标
        
        Args:
            propensity_scores: 倾向性得分 [n_samples]
            treatment_groups: 治疗组标签 [n_samples]
        
        Returns:
            balance_metrics: 平衡性指标字典
        """
        if propensity_scores is None or len(torch.unique(treatment_groups)) < 2:
            return {'standardized_mean_diff': 0.0, 'variance_ratio': 1.0}
        
        try:
            # 计算不同组的倾向性得分统计量
            unique_groups = torch.unique(treatment_groups)
            group_stats = {}
            
            for group in unique_groups:
                mask = (treatment_groups == group)
                if mask.sum() > 0:
                    group_scores = propensity_scores[mask]
                    group_stats[group.item()] = {
                        'mean': group_scores.mean().item(),
                        'var': group_scores.var().item(),
                        'n': mask.sum().item()
                    }
            
            if len(group_stats) >= 2:
                groups = list(group_stats.keys())
                
                # 标准化均值差异
                mean_diff = abs(group_stats[groups[0]]['mean'] - group_stats[groups[1]]['mean'])
                pooled_var = (group_stats[groups[0]]['var'] + group_stats[groups[1]]['var']) / 2
                standardized_mean_diff = mean_diff / (np.sqrt(pooled_var) + 1e-8)
                
                # 方差比率
                var_ratio = max(group_stats[groups[0]]['var'], group_stats[groups[1]]['var']) / \
                           (min(group_stats[groups[0]]['var'], group_stats[groups[1]]['var']) + 1e-8)
                
                return {
                    'standardized_mean_diff': standardized_mean_diff,
                    'variance_ratio': var_ratio
                }
            
        except Exception as e:
            print(f"Warning: Error computing propensity score balance: {e}")
        
        return {'standardized_mean_diff': 0.0, 'variance_ratio': 1.0}


def compute_causal_effects(model, test_loader, clinical_features_baseline, device='cuda'):
    """
    计算因果效应（用于模型评估）
    
    Args:
        model: 训练好的因果模型
        test_loader: 测试数据加载器
        clinical_features_baseline: 基线临床特征
        device: 计算设备
    
    Returns:
        causal_effects: 因果效应字典
    """
    model.eval()
    
    all_predictions = []
    all_counterfactual_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取原始预测
            original_predictions = model(**batch)
            all_predictions.append(original_predictions)
            
            # 生成反事实预测（修改临床特征）
            if 'clinical_features' in batch:
                modified_batch = batch.copy()
                modified_batch['clinical_features'] = clinical_features_baseline.to(device)
                counterfactual_predictions = model(**modified_batch)
                all_counterfactual_predictions.append(counterfactual_predictions)
    
    if len(all_counterfactual_predictions) > 0:
        # 计算平均因果效应
        original_preds = torch.cat(all_predictions, dim=0)
        counterfactual_preds = torch.cat(all_counterfactual_predictions, dim=0)
        
        # 计算风险差异
        original_risks = torch.sigmoid(original_preds).sum(dim=1)
        counterfactual_risks = torch.sigmoid(counterfactual_preds).sum(dim=1)
        
        average_treatment_effect = (original_risks - counterfactual_risks).mean().item()
        
        return {
            'average_treatment_effect': average_treatment_effect,
            'effect_heterogeneity': (original_risks - counterfactual_risks).std().item()
        }
    
    return {'average_treatment_effect': 0.0, 'effect_heterogeneity': 0.0}
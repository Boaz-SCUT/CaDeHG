"""
多模态平衡学习模块 for SurvPath
实现OGM (On-the-fly Gradient Modulation) 方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BalancedMultimodalLearning(nn.Module):
    """
    实现多模态平衡学习的OGM策略
    在SurvPath中平衡WSI和基因表达数据的学习
    """
    
    def __init__(self, n_classes=4, modulation_starts=5, modulation_ends=100, alpha=0.5):
        """
        Args:
            n_classes: 分类数量（对于生存预测，是时间bins数量）
            modulation_starts: 从哪个epoch开始进行梯度调制
            modulation_ends: 在哪个epoch结束梯度调制
            alpha: 调制强度参数
        """
        super(BalancedMultimodalLearning, self).__init__()
        self.n_classes = n_classes
        self.modulation_starts = modulation_starts
        self.modulation_ends = modulation_ends
        self.alpha = alpha
        
        # 用于评估每个模态性能的分类器
        self.pathway_classifier = nn.Linear(256, n_classes)  # pathway embedding维度
        self.wsi_classifier = nn.Linear(256, n_classes)  # WSI embedding维度
        
    def forward_unimodal_predictions(self, pathway_embed, wsi_embed):
        """
        获取每个模态的单独预测
        
        Args:
            pathway_embed: pathway模态的embeddings [B, 256]
            wsi_embed: WSI模态的embeddings [B, 256]
            
        Returns:
            pathway_logits: pathway模态预测 [B, n_classes]
            wsi_logits: WSI模态预测 [B, n_classes]
        """
        pathway_logits = self.pathway_classifier(pathway_embed)
        wsi_logits = self.wsi_classifier(wsi_embed)
        return pathway_logits, wsi_logits
    
    def calculate_discrepancy_ratio(self, pathway_logits, wsi_logits, labels):
        """
        计算两个模态之间的判别性差异比率
        
        Args:
            pathway_logits: pathway预测 [B, n_classes]
            wsi_logits: WSI预测 [B, n_classes]
            labels: 真实标签 [B]
            
        Returns:
            rho_pathway: pathway模态的差异比率
            rho_wsi: WSI模态的差异比率
        """
        # 转换为概率
        pathway_probs = F.softmax(pathway_logits, dim=-1)
        wsi_probs = F.softmax(wsi_logits, dim=-1)
        
        # 获取正确类别的概率
        batch_size = labels.shape[0]
        labels = labels.squeeze().long()
        
        pathway_correct_probs = pathway_probs[range(batch_size), labels]
        wsi_correct_probs = wsi_probs[range(batch_size), labels]
        
        # 计算平均概率
        avg_pathway_prob = pathway_correct_probs.mean()
        avg_wsi_prob = wsi_correct_probs.mean()
        
        # 计算差异比率
        epsilon = 1e-8
        rho_pathway = avg_pathway_prob / (avg_wsi_prob + epsilon)
        rho_wsi = avg_wsi_prob / (avg_pathway_prob + epsilon)
        
        return rho_pathway, rho_wsi
    
    def calculate_modulation_coefficients(self, rho_pathway, rho_wsi):
        """
        根据差异比率计算梯度调制系数
        
        Args:
            rho_pathway: pathway模态的差异比率
            rho_wsi: WSI模态的差异比率
            
        Returns:
            k_pathway: pathway的调制系数
            k_wsi: WSI的调制系数
        """
        # 如果某个模态表现更好（rho > 1），则降低其梯度
        k_pathway = 1.0 - self.alpha * torch.clamp(rho_pathway - 1.0, min=0)
        k_wsi = 1.0 - self.alpha * torch.clamp(rho_wsi - 1.0, min=0)
        
        return k_pathway, k_wsi
    
    def apply_gradient_modulation(self, model, k_pathway, k_wsi, current_epoch):
        """
        应用梯度调制到模型参数
        
        Args:
            model: SurvPath模型
            k_pathway: pathway的调制系数
            k_wsi: WSI的调制系数
            current_epoch: 当前epoch数
        """
        # 只在指定的epoch范围内进行调制
        if current_epoch < self.modulation_starts or current_epoch > self.modulation_ends:
            return
        
        # 对pathway编码器应用调制
        for param in model.sig_networks.parameters():
            if param.grad is not None:
                param.grad *= k_pathway
        
        # 对WSI投影网络应用调制
        for param in model.wsi_projection_net.parameters():
            if param.grad is not None:
                param.grad *= k_wsi


class BalancedSurvPathTrainer:
    """
    集成了多模态平衡学习的SurvPath训练器
    """
    
    def __init__(self, model, balanced_module, optimizer, loss_fn, device):
        """
        Args:
            model: SurvPath模型
            balanced_module: BalancedMultimodalLearning模块
            optimizer: 优化器
            loss_fn: 损失函数
            device: 设备
        """
        self.model = model
        self.balanced_module = balanced_module
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    def train_step(self, data_WSI, data_omics, y_disc, event_time, censor, mask, current_epoch):
        """
        单次训练步骤，集成了梯度调制
        
        Args:
            data_WSI: WSI数据
            data_omics: 基因表达数据列表
            y_disc: 离散化标签
            event_time: 事件时间
            censor: 审查状态
            mask: mask
            current_epoch: 当前epoch
            
        Returns:
            loss: 损失值
            risk: 风险评分
        """
        self.optimizer.zero_grad()
        
        # 前向传播 - 获取中间表示
        # 1. 处理pathway embeddings
        h_omic = [self.model.sig_networks[idx].forward(sig_feat.float()) 
                  for idx, sig_feat in enumerate(data_omics)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)  # [1, num_pathways, 256]
        
        # 2. 处理WSI embeddings
        wsi_embed = self.model.wsi_projection_net(data_WSI)  # [num_patches, 256] or [B, num_patches, 256]
        
        # 3. 获取模态特定的聚合表示（用于评估单模态性能）
        # 确保正确处理维度
        if h_omic_bag.dim() == 3:  # [B, num_pathways, 256]
            pathway_mean_embed = torch.mean(h_omic_bag, dim=1)  # [B, 256]
        else:
            pathway_mean_embed = h_omic_bag
        
        # WSI embeddings聚合：在patch维度上求平均
        if wsi_embed.dim() == 3:  # [B, num_patches, 256]
            wsi_mean_embed = torch.mean(wsi_embed, dim=1)  # [B, 256]
        elif wsi_embed.dim() == 2:  # [num_patches, 256]，缺少batch维度
            wsi_mean_embed = torch.mean(wsi_embed, dim=0, keepdim=True)  # [1, 256]
        else:
            wsi_mean_embed = wsi_embed
        
        # 4. 获取单模态预测（用于计算差异）
        pathway_logits, wsi_logits = self.balanced_module.forward_unimodal_predictions(
            pathway_mean_embed, wsi_mean_embed
        )
        
        # 5. 完整的多模态前向传播
        input_args = {"x_path": data_WSI}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(self.device)
        input_args["return_attn"] = False
        
        logits = self.model(**input_args)
        
        # 计算主要损失
        loss = self.loss_fn(h=logits, y=y_disc, t=event_time, c=censor)
        
        # 添加辅助损失（用于训练单模态分类器）
        # 注意：使用squeeze(-1)而不是squeeze()，避免batch_size=1时维度被压缩掉
        target = y_disc.squeeze(-1).long() if y_disc.dim() > 1 else y_disc.long()
        aux_loss_pathway = F.cross_entropy(pathway_logits, target)
        aux_loss_wsi = F.cross_entropy(wsi_logits, target)
        auxiliary_loss = 0.1 * (aux_loss_pathway + aux_loss_wsi)  # 权重较小
        
        total_loss = loss + auxiliary_loss
        
        # 反向传播
        total_loss.backward()
        
        # 计算并应用梯度调制
        with torch.no_grad():
            rho_pathway, rho_wsi = self.balanced_module.calculate_discrepancy_ratio(
                pathway_logits.detach(), wsi_logits.detach(), y_disc
            )
            k_pathway, k_wsi = self.balanced_module.calculate_modulation_coefficients(
                rho_pathway, rho_wsi
            )
            
            # print(f"Modality balance - Pathway: {rho_pathway:.3f}, WSI: {rho_wsi:.3f}, "
            #       f"k_pathway: {k_pathway:.3f}, k_wsi: {k_wsi:.3f}")
        
        # 应用梯度调制
        self.balanced_module.apply_gradient_modulation(
            self.model, k_pathway, k_wsi, current_epoch
        )
        
        # 优化器步骤
        self.optimizer.step()
        
        # 计算风险
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        
        return total_loss.item(), risk, rho_pathway.item(), rho_wsi.item()


class OPMBalancedLearning(nn.Module):
    """
    实现OPM (On-the-fly Prediction Modulation) 策略
    在前向传播阶段通过dropout来调制模态
    """
    
    def __init__(self, alpha=0.5):
        super(OPMBalancedLearning, self).__init__()
        self.alpha = alpha
        
    def calculate_drop_prob(self, rho):
        """
        根据差异比率计算dropout概率
        
        Args:
            rho: 模态的差异比率
            
        Returns:
            drop_prob: dropout概率
        """
        if rho > 1:  # 该模态表现更好
            drop_prob = self.alpha * (rho - 1.0) / rho
        else:
            drop_prob = 0.0
        return min(drop_prob, 0.5)  # 限制最大dropout率
    
    def apply_modulation(self, embeddings, drop_prob, training=True):
        """
        应用dropout调制
        
        Args:
            embeddings: 模态embeddings
            drop_prob: dropout概率
            training: 是否在训练模式
            
        Returns:
            调制后的embeddings
        """
        if training and drop_prob > 0:
            mask = torch.bernoulli(torch.ones_like(embeddings) * (1 - drop_prob))
            embeddings = embeddings * mask / (1 - drop_prob)  # scaled dropout
        return embeddings
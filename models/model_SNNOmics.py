from collections import OrderedDict
from os.path import join
import pdb
from transformers import AutoTokenizer, AutoModel
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

"""

Implement a self normalizing network to handle tabular omics data with multi-task learning (survival + stage prediction)

Klambauer, Günter, et al. "Self-normalizing neural networks." Advances in neural information processing systems 30 (2017).

"""
class GatedFusion(nn.Module):
    """门控融合机制"""
    def __init__(self, omic_dim, text_dim):
        super(GatedFusion, self).__init__()
        self.common_dim = max(omic_dim, text_dim)
        
        self.omic_proj = nn.Linear(omic_dim, self.common_dim) if omic_dim != self.common_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, self.common_dim) if text_dim != self.common_dim else nn.Identity()
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(omic_dim + text_dim, omic_dim + text_dim),
            nn.Sigmoid()
        )
        
    def forward(self, omic_feat, text_feat):
        # 投影到相同维度
        # omic_proj = self.omic_proj(omic_feat)
        # text_proj = self.text_proj(text_feat)
        
        # 计算门控权重
        combined = torch.cat([omic_feat, text_feat], dim=1)
        gate_weights = self.gate(combined)
        
        # 门控融合
        # fused = gate_weights * omic_feat + (1 - gate_weights) * text_feat
        fused = torch.cat([gate_weights*omic_feat, (1 - gate_weights)*text_feat], dim=1)  # 保留原始特征

        return fused

##########################
#### Genomic FC Model ####
##########################
class SNNOmics(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4, 
                 n_stage_classes: int=4, enable_multitask: bool=False, multitask_weight: float=0.3):
        super(SNNOmics, self).__init__()
        self.n_classes = n_classes
        self.n_stage_classes = n_stage_classes
        self.enable_multitask = enable_multitask
        self.multitask_weight = multitask_weight
        
        # self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        self.size_dict_omic = {'small': [768, 768], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        
        # Survival prediction head
        # self.survival_classifier = nn.Linear(hidden[-1]+768, n_classes)
        # self.survival_classifier = nn.Linear(hidden[-1], n_classes)
        
        self.survival_classifier = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-1]//2),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(hidden[-1]//2, n_classes)
        )
        
        # Stage prediction head (only if multitask is enabled)
        if self.enable_multitask:
            self.stage_classifier = nn.Linear(hidden[-1], n_stage_classes)
        
        # biobert 编码 text report
        self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained("/data/TCGA/biobert")
        self.clinical_bert_model = AutoModel.from_pretrained("/data/TCGA/biobert")
        
        self.text_classifier = nn.Linear(768,n_classes)
        self.gene_classifier = nn.Linear(hidden[-1], n_classes)
        self.constant = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.debiase_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.fusion_lin = nn.Linear(2*n_classes, n_classes)
        # self.fusion_lin = nn.Sequential(
        #     nn.Linear(2*n_classes, n_classes),
        #     nn.ReLU(),
        #     # nn.Dropout(0.25), 
        #     nn.Linear(n_classes, n_classes)   
        # )
        self.fusion_lin_add = nn.Sequential(
            nn.Linear(n_classes, n_classes),
            nn.ReLU(),
            # nn.Dropout(0.25),
        )
        
        self.text_classifier_stage = nn.Linear(768, n_stage_classes)
        self.gene_classifier_stage = nn.Linear(hidden[-1], n_stage_classes)
        self.debiase_weight2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        init_max_weights(self)

    def fusion(self, z_main, z_text, z_gene, main_fact=False, text_fact=False, gene_fact=False):
        # Apply counterfactual transformations
        if not main_fact:
            z_main = self.constant * torch.ones_like(z_main).cuda()
        if not text_fact:
            z_text = self.constant * torch.ones_like(z_text).cuda()
        if not gene_fact:
            z_gene = self.constant * torch.ones_like(z_gene).cuda()
        
        # # Apply counterfactual transformations
        # if not main_fact:
        #     z_main = torch.ones_like(z_main).cuda()
        # if not text_fact:
        #     z_text = torch.ones_like(z_text).cuda()
        # if not gene_fact:
        #     z_gene = torch.ones_like(z_gene).cuda()
            
        # cfvqa
        z = z_main + z_text
        if z_gene is not None and gene_fact:
            z = z + z_gene
        z = torch.log(torch.sigmoid(z) + 1e-9)
        # z = torch.nn.functional.leaky_relu(z) 
        # z = torch.log(torch.nn.functional.relu(z) + 1e-9)  # ReLU activation
        
        
        # # cat + linear fusion
        # z = torch.cat((z_main, z_text), dim=1)
        # if z_gene is not None and gene_fact:
        #     z = torch.cat((z, z_gene), dim=1)
        # z = self.fusion_lin(z)
        
        # # add + linear fusion
        # z = z_main + z_text
        # if z_gene is not None and gene_fact:
        #     z = z + z_gene
        # z = self.fusion_lin_add(z)
        
        return z
    
    def fusion_stage(self, z_main, z_text, z_gene, main_fact=False, text_fact=False, gene_fact=False):
        # Apply counterfactual transformations
        if not main_fact:
            z_main = self.constant * torch.ones_like(z_main).cuda()
        if not text_fact:
            z_text = self.constant * torch.ones_like(z_text).cuda()
        if not gene_fact:
            z_gene = self.constant * torch.ones_like(z_gene).cuda()
            
        # cfvqa
        z = z_main + z_text
        if z_gene is not None and gene_fact:
            z = z + z_gene
        # z = torch.log(torch.sigmoid(z) + 1e-9)
        z = torch.nn.functional.leaky_relu(z) 
        
        # # cat + linear fusion
        # z = torch.cat((z_main, z_text), dim=1)
        # if z_gene is not None and gene_fact:
        #     z = torch.cat((z, z_gene), dim=1)
        # z = self.fusion_lin(z)
        
        # # add + linear fusion
        # z = z_main + z_text
        # if z_gene is not None and gene_fact:
        #     z = z + z_gene
        # z = self.fusion_lin_add(z)
        
        return z
    def forward(self, return_feats=False, **kwargs):
        x = kwargs['data_omics']
        h_omic = self.fc_omic(x)
        # x_text_report = kwargs['x_text_report'][-1]
        # x_text_report = 'This is the pathology report of this patient, please read and analyze the important information which is related to the survival of this patient: ' + x_text_report
        # print('x_text_report: ', x_text_report)
        # x_text_report = 'The pathology report of this patient is: ' + x_text_report
        # text_inputs = self.clinical_bert_tokenizer(
        #     x_text_report, 
        #     return_tensors="pt", 
        #     truncation=True, 
        #     padding=True, 
        #     max_length=512
        # ).to(h_omic.device)
        
        # outputs = self.clinical_bert_model(**text_inputs)
        # 使用[CLS] token的嵌入
        # text_embeddings = outputs.last_hidden_state[:, 0, :]

        # cat_embeddings = torch.cat((h_omic, text_embeddings), dim=1)
        
        
        # Survival prediction
        # survival_logits = self.survival_classifier(cat_embeddings)
        survival_logits = self.survival_classifier(h_omic)
        assert len(survival_logits.shape) == 2 and survival_logits.shape[1] == self.n_classes
        
        if self.enable_multitask:
            # Stage prediction
            stage_logits = self.stage_classifier(h_omic)
            assert len(stage_logits.shape) == 2 and stage_logits.shape[1] == self.n_stage_classes
            
            if return_feats:
                return h_omic, survival_logits, stage_logits
            return survival_logits, stage_logits
        else:
            if return_feats:
                return h_omic, survival_logits
            return survival_logits
        
    def forward1(self, return_feats=False, **kwargs):
        '''先对齐，再融合'''
        x = kwargs['data_omics']
        h_omic = self.fc_omic(x)
        x_text_report = kwargs['x_text_report'][-1]
        # x_text_report = 'This is the pathology report of this patient, please read and analyze the important information which is related to the survival of this patient: ' + x_text_report
        # print('x_text_report: ', x_text_report)
        x_text_report = 'The pathology report of this patient is: ' + x_text_report
        text_inputs = self.clinical_bert_tokenizer(
            x_text_report, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(h_omic.device)
        
        outputs = self.clinical_bert_model(**text_inputs)
        # 使用[CLS] token的嵌入
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 特征对齐

        cat_embeddings = torch.cat((h_omic, text_embeddings), dim=1)
        
        
        # Survival prediction
        survival_logits = self.survival_classifier(cat_embeddings)
        # survival_logits = self.survival_classifier(h_omic)
        assert len(survival_logits.shape) == 2 and survival_logits.shape[1] == self.n_classes
        
        if self.enable_multitask:
            # Stage prediction
            stage_logits = self.stage_classifier(h_omic)
            assert len(stage_logits.shape) == 2 and stage_logits.shape[1] == self.n_stage_classes
            
            if return_feats:
                return h_omic, survival_logits, stage_logits
            return survival_logits, stage_logits
        else:
            if return_feats:
                return h_omic, survival_logits
            return survival_logits

  
    def forward0(self, return_feats=False, **kwargs):
        
        x = kwargs['data_omics']
        h_omic = self.fc_omic(x)
        x_text_report = kwargs['x_text_report'][-1]
        # x_text_report = 'This is the pathology report of this patient, please read and analyze the important information which is related to the survival of this patient: ' + x_text_report
        # print('x_text_report: ', x_text_report)
        x_text_report = 'The pathology report is: ' + x_text_report
        text_inputs = self.clinical_bert_tokenizer(
            x_text_report, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(h_omic.device)
        
        outputs = self.clinical_bert_model(**text_inputs)
        # 使用[CLS] token的嵌入
        text_embeddings = outputs.last_hidden_state[:, 0, :]

        cat_embeddings = torch.cat((h_omic, text_embeddings), dim=1)
        
        # Survival prediction
        survival_logits = self.survival_classifier(cat_embeddings)
        # survival_logits = self.survival_classifier(h_omic)
        assert len(survival_logits.shape) == 2 and survival_logits.shape[1] == self.n_classes
        
        # logits_text
        logits_text = self.text_classifier(text_embeddings)
        logits_gene = self.gene_classifier(h_omic)
        # TE 
        logit_te = self.fusion(survival_logits, logits_text, logits_gene,
                                main_fact=True, text_fact=True, gene_fact=False)
        logits_nde = self.fusion(survival_logits.clone().detach(), logits_text.clone().detach(), logits_gene,
                            main_fact=False, text_fact=True, gene_fact=False) # NDE
        # nde_weight = torch.sigmoid(self.nde_weight)
        # nde_weight = torch.sigmoid(self.debiase_weight)
        # logit_te = logit_te - nde_weight * logits_nde
        
        logits_tie = logit_te - 0.5 * logits_nde
                
        if self.enable_multitask:
            # Stage prediction
            stage_logits = self.stage_classifier(h_omic)
            assert len(stage_logits.shape) == 2  and stage_logits.shape[1] == self.n_stage_classes
            logits_text_stage = self.text_classifier_stage(text_embeddings)
            gene_logits_stage = self.gene_classifier_stage(h_omic)
            # logit_te_stage = self.fusion_stage(stage_logits, text_logits_stage, gene_logits_stage,
            #                                     main_fact=True, text_fact=True, gene_fact=False)
            # logits_nde_stage = self.fusion_stage(stage_logits.clone().detach(), text_logits_stage.clone().detach(), gene_logits_stage,
            #                                     main_fact=False, text_fact=True, gene_fact=False) # NDE
            # nde_weight_stage2 = torch.sigmoid(self.debiase_weight2)
            # logit_te_stage = logit_te_stage + nde_weight_stage2 * logits_nde_stage
            
            if return_feats:
                return h_omic, survival_logits, stage_logits, logits_text, logit_te
            return survival_logits, stage_logits, logits_text, logit_te
        else:
            if return_feats:
                return h_omic, survival_logits
            return survival_logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
        else:
            self.fc_omic = self.fc_omic.to(device)

        self.survival_classifier = self.survival_classifier.to(device)
        if self.enable_multitask:
            self.stage_classifier = self.stage_classifier.to(device)

def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
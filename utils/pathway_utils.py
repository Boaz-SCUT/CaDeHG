# utils/pathway_utils.py
import torch
import pandas as pd
import numpy as np

def build_pathway_gene_matrix(omic_names, all_gene_names):
    """
    构造通路-基因关系矩阵
    
    Args:
        omic_names: 每个通路包含的基因列表 (长度=331)
        all_gene_names: 所有基因的名称列表 (长度=4999)
    
    Returns:
        pathway_gene_matrix: [331, 4999] 的二进制矩阵
    """
    num_pathways = len(omic_names)
    num_genes = len(all_gene_names)
    
    print(f"构造通路-基因矩阵: {num_pathways}个通路 × {num_genes}个基因")
    
    # 创建基因名到索引的映射
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_gene_names)}
    
    # 初始化矩阵
    pathway_gene_matrix = torch.zeros(num_pathways, num_genes)
    
    # 填充矩阵
    for pathway_idx, pathway_genes in enumerate(omic_names):
        genes_found = 0
        for gene in pathway_genes:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                pathway_gene_matrix[pathway_idx, gene_idx] = 1.0
                genes_found += 1
        
        # print(f"通路 {pathway_idx}: {len(pathway_genes)}个基因定义, {genes_found}个基因匹配")
    
    print(f"通路-基因矩阵形状: {pathway_gene_matrix.shape}")
    print(f"非零元素比例: {torch.mean(pathway_gene_matrix).item():.4f}")
    print(f"每个通路平均基因数: {torch.mean(torch.sum(pathway_gene_matrix, dim=1)).item():.1f}")
    
    return pathway_gene_matrix
def get_pathway_prior_probs(pathway_names):
    """
    根据通路的生物学重要性设置先验概率
    """
    # 可以根据专业知识调整
    important_pathways = [
        'CELL_CYCLE', 'APOPTOSIS', 'DNA_REPAIR', 'IMMUNE_RESPONSE', 
        'PROLIFERATION', 'METASTASIS'
    ]
    
    priors = []
    for name in pathway_names:
        if any(imp in name.upper() for imp in important_pathways):
            priors.append(0.15)  # 重要通路高权重
        else:
            priors.append(0.05)  # 其他通路低权重
    
    priors = torch.tensor(priors)
    priors = priors / priors.sum()  # 归一化
    
    return priors
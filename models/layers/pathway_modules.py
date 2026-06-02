import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayGraphBuilder(nn.Module):
    def __init__(self, num_pathways, pathway_dim, correlation_weight=0.3, 
                 overlap_weight=0.3, prior_weight=0.4):
        super(PathwayGraphBuilder, self).__init__()
        
        self.num_pathways = num_pathways
        self.pathway_dim = pathway_dim
        self.weights = [correlation_weight, overlap_weight, prior_weight]
        
        # 计算相关性矩阵的映射层
        self.correlation_mapper = nn.Sequential(
            nn.Linear(pathway_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # 加载先验知识 (实际应用中应从数据库加载)
        self.register_buffer(
            'prior_knowledge', 
            torch.zeros(num_pathways, num_pathways)
        )
        
        # 加载基因重叠矩阵 (实际应用中应从数据库加载)
        self.register_buffer(
            'gene_overlap', 
            torch.zeros(num_pathways, num_pathways)
        )
    
    def forward(self, pathway_activities):
        batch_size = pathway_activities.shape[0]
        
        # 计算相关性矩阵
        correlation_matrix = self._compute_correlation_matrix(pathway_activities)
        
        # 整合三种关系来源
        combined_graph = (
            self.weights[0] * correlation_matrix 
            # self.weights[1] * self.gene_overlap + 
            # self.weights[2] * self.prior_knowledge
        )
        
        return combined_graph
    
    def _compute_correlation_matrix(self, pathway_activities):
        batch_size, num_pathways, dim = pathway_activities.shape
        corr_matrix = torch.zeros(batch_size, num_pathways, num_pathways)
        
        for i in range(num_pathways):
            for j in range(num_pathways):
                if i == j:
                    corr_matrix[:, i, j] = 1.0
                    continue
                    
                # 将两个通路的活性拼接
                paired_features = torch.cat(
                    [pathway_activities[:, i, :], pathway_activities[:, j, :]], 
                    dim=1
                )
                
                # 计算相关性得分
                corr_matrix[:, i, j] = self.correlation_mapper(paired_features).squeeze(-1)
        
        return corr_matrix


class PathwayInteractionEncoder(nn.Module):
    def __init__(self, num_pathways, pathway_dim, interaction_dim, num_gcn_layers=2):
        super(PathwayInteractionEncoder, self).__init__()
        
        self.num_pathways = num_pathways
        self.pathway_dim = pathway_dim
        self.interaction_dim = interaction_dim
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            input_dim = pathway_dim if i == 0 else interaction_dim
            self.gcn_layers.append(GCNLayer(input_dim, interaction_dim))
        
        # 输出映射层
        self.output_layer = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.LeakyReLU()
        )
    
    def forward(self, pathway_activities, pathway_graph):
        batch_size = pathway_activities.shape[0]
        
        # 初始特征
        x = pathway_activities
        
        # 应用GCN层
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, pathway_graph)
        
        # 通过全局池化获取图级表示
        global_features = torch.mean(x, dim=1)  # [B, dim]
        
        # 输出映射
        output = self.output_layer(global_features)
        
        return output


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        # x: [B, N, F]
        # adj: [B, N, N]
        
        batch_size, num_nodes, _ = x.shape
        
        # 归一化邻接矩阵
        adj_norm = self._normalize_adj(adj).to(x.device)
        
        # 图卷积操作
        support = torch.matmul(x, self.weight).to(x.device)  # [B, N, F']
        output = torch.matmul(adj_norm, support)  # [B, N, F']
        output = output + self.bias
        
        return F.leaky_relu(output)
    
    def _normalize_adj(self, adj):
        # 计算度矩阵
        degree = torch.sum(adj, dim=2, keepdim=True)
        degree = torch.clamp(degree, min=1e-6)
        
        # D^(-1/2) * A * D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        adj_norm = degree_inv_sqrt * adj * degree_inv_sqrt.transpose(1, 2)
        
        return adj_norm
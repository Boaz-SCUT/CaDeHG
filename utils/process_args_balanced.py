"""
修改后的参数处理脚本
添加了多模态平衡学习相关的参数
"""

import argparse


def _process_args():
    """
    处理命令行参数，包含平衡学习相关参数
    
    Returns:
        args: argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description='Configurations for SurvPath with Balanced Multimodal Learning'
    )

    # ========== 原有的SurvPath参数 ==========
    
    # 研究相关
    parser.add_argument('--study', type=str, help='study name')
    parser.add_argument('--task', type=str, choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="hallmarks", 
                       choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    # 数据相关
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default=None, 
                       help='Path to dir with omics csv for all modalities')
    parser.add_argument('--num_patches', type=int, default=4000, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months_dss", 
                       help='type of survival (OS, DSS, PFI)')
    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)

    # 分割相关
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default=None, 
                       help='manually specify the set of splits to use')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')
        
    # 训练相关
    parser.add_argument('--max_epochs', type=int, default=20, 
                       help='maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, 
                       help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, 
                       help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, 
                       choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv"], 
                       default='nll_surv',
                       help='survival loss function (default: nll_surv)')
    parser.add_argument('--alpha_surv', type=float, default=0.0, 
                       help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)

    # 模型相关
    parser.add_argument('--fusion', type=str, default=None)
    parser.add_argument('--modality', type=str, default="survpath")
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')
    parser.add_argument('--use_nystrom', action='store_true', default=False, 
                       help='Use Nystrom attention in SurvPath.')

    # ========== 新增：多模态平衡学习参数 ==========
    
    parser.add_argument('--use_balanced_learning', action='store_true', default=True,
                       help='Enable balanced multimodal learning (OGM method)')
    
    parser.add_argument('--balance_method', type=str, default='ogm', 
                       choices=['ogm', 'opm', 'ogm_opm'],
                       help='Balanced learning method: ogm (gradient modulation), ' 
                            'opm (prediction modulation), or ogm_opm (both)')
    
    parser.add_argument('--balance_alpha', type=float, default=0.5,
                       help='Modulation strength for balanced learning (default: 0.5). '
                            'Higher values = stronger modulation')
    
    parser.add_argument('--modulation_starts', type=int, default=5,
                       help='Start applying gradient modulation from this epoch (default: 5)')
    
    parser.add_argument('--modulation_ends', type=int, default=100,
                       help='Stop applying gradient modulation at this epoch (default: 100)')
    
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                       help='Weight for auxiliary unimodal classification losses (default: 0.1)')
    
    parser.add_argument('--log_modality_balance', action='store_true', default=True,
                       help='Log modality balance metrics during training')
    
    parser.add_argument('--adaptive_alpha', action='store_true', default=False,
                       help='Adaptively adjust alpha based on modality discrepancy. '
                            'If enabled, alpha increases when imbalance is larger')
    
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (default: 10 epochs)')

    # 可视化和分析
    parser.add_argument('--save_attention_maps', action='store_true', default=False,
                       help='Save attention maps for analysis')
    
    parser.add_argument('--visualize_balance', action='store_true', default=False,
                       help='Create visualizations of modality balance over training')

    args = parser.parse_args()

    # 验证参数
    if not (args.task == "survival"):
        print("Task must be 'survival'")
        exit()
    
    if args.modulation_starts >= args.modulation_ends:
        print("Warning: modulation_starts should be less than modulation_ends")
        print(f"Setting modulation_ends to {args.modulation_starts + 10}")
        args.modulation_ends = args.modulation_starts + 10
    
    if args.balance_alpha < 0 or args.balance_alpha > 1:
        print("Warning: balance_alpha should be between 0 and 1")
        args.balance_alpha = max(0, min(1, args.balance_alpha))
        print(f"Adjusted balance_alpha to {args.balance_alpha}")

    return args

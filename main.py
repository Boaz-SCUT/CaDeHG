# debug 指定GPU
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#----> pytorch imports
import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告
import torch
import gc


#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args

def main0(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    for i in folds:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _train_val(datasets, i, args)

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)
    
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        _save_pkl(filename, results)
    
    final_df = pd.DataFrame({
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        'val_BS': all_val_BS,
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))

def main(args):
    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    for i in folds:
        # i += 4
        # if i > 4:
        #     exit(0)
        print(f"=== Starting Fold {i} ===")
        # 在每个fold开始前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before fold {i}: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        # 根据模型类型选择训练函数
        if args.modality == "ccd_survpath":
            from utils.core_utils import _train_val_ccd
            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _train_val_ccd(datasets, i, args)
        else:
            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _train_val(datasets, i, args)

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)
    
        #write results to pkl
        # filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        # _save_pkl(filename, results)
        
        current_folds = folds[:len(all_val_cindex)]  # 已完成的fold
        current_df = pd.DataFrame({
            'folds': current_folds,
            'val_cindex': all_val_cindex,
            'val_cindex_ipcw': all_val_cindex_ipcw,
            'val_IBS': all_val_IBS,
            'val_iauc': all_val_iauc,
            "val_loss": all_val_loss,
            'val_BS': all_val_BS,
        })
        
        # 动态确定保存文件名
        if len(all_val_cindex) == len(folds):
            summary_name = 'summary.csv'  # 所有fold完成
        else:
            summary_name = f'summary_partial_{len(all_val_cindex)}folds.csv'  # 部分完成
            
        current_df.to_csv(os.path.join(args.results_dir, summary_name), index=False)
        
        print(f"Updated summary after fold {i}: {len(all_val_cindex)}/{len(folds)} folds completed")
        
        
        
        # === 关键：在每个fold结束后进行显存清理 ===
        print(f"=== Cleaning up after Fold {i} ===")
        
        # 删除数据集引用
        del datasets
        del results
        
        # 强制垃圾回收
        gc.collect()
        
        # 清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after cleanup fold {i}: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    final_df = pd.DataFrame({
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        'val_BS': all_val_BS,
    })

    if len(folds) != args.k:
        start, end = folds[0], folds[-1]
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    # 输出每个fold的最佳c-index
    print("=== Final Results Summary ===")
    for i, cindex in enumerate(all_val_cindex):
        print(f"Fold {i+1}: C-index = {cindex:.4f}")
    
    # 打印CCD效果总结
    # if args.modality == "ccd_survpath":
    print("=== CCD效果总结 ===")
    print(f"平均C-index: {np.mean(all_val_cindex):.4f} ± {np.std(all_val_cindex):.4f}")
    print(f"平均IBS: {np.mean(all_val_IBS):.4f} ± {np.std(all_val_IBS):.4f}")
    print(f"平均iAUC: {np.mean(all_val_iauc):.4f} ± {np.std(all_val_iauc):.4f}")

if __name__ == "__main__":
    start = timer()
    import sys
    # 手动设置命令行参数
    # sys.argv = [
    #     'main.py',
    #     '--study', 'tcga_blca',
    #     '--task', 'survival',
    #     '--split_dir', 'splits',
    #     '--which_splits', '5foldcv',
    #     '--type_of_path', 'combine',
    #     '--modality', 'snn',
    #     '--data_root_dir', '/data/TCGA/BLCA/features/pt_files/',
    #     '--label_file', 'datasets_csv/metadata/tcga_blca.csv',
    #     '--omics_dir', 'datasets_csv/raw_rna_data/combine/blca',
    #     '--results_dir', 'results_blca',
    #     '--batch_size', '1',
    #     '--lr', '0.00005',
    #     '--opt', 'adam',
    #     '--reg', '0.00001',
    #     '--alpha_surv', '0.5',
    #     '--weighted_sample',
    #     '--max_epochs', '5',
    #     '--encoding_dim', '256',
    #     '--label_col', 'survival_months_dss',
    #     '--k', '5',
    #     '--bag_loss', 'nll_surv',
    #     '--n_classes', '4',
    #     '--num_patches', '4090',
    #     '--wsi_projection_dim', '256',
    #     '--encoding_layer_1_dim', '8',
    #     '--encoding_layer_2_dim', '16',
    #     '--encoder_dropout', '0.25'
    # ]
    # sys.argv = [
    #     'main.py',
    #     '--study', 'tcga_blca',
    #     '--task', 'survival',
    #     '--modality', 'ccd_survpath',
    #     '--type_of_path', 'combine',
    #     '--n_classes', '4',
    #     '--max_epochs', '1',
    #     '--lr', '1e-4',
    #     '--alpha_ccd', '1.0',
    #     '--beta_ccd', '0.5',
    #     '--batch_size', '1',
    #     '--label_col', 'survival_months_dss',
    #     '--data_root_dir', '/data/TCGA/BLCA/features/pt_files/',
    #     '--label_file', 'datasets_csv/metadata/tcga_blca.csv',
    #     '--omics_dir', 'datasets_csv/raw_rna_data/combine/blca',
    #     '--results_dir', 'ccd_blca_results',
    #     '--which_splits', '5foldcv',
    #     '--seed', '42',
    #     '--wsi_projection_dim', '256',
    #     '--num_patches', '4090',
    #     '--use_counterfactual',
    #     '--bag_loss', 'nll_surv'
    # ]
    

    #----> read the args
    args = _process_args()
    
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality in ["survpath", "ccd_survpath", "double_mediator_survpath"] else False, # 改 
        type_of_pathway=args.type_of_path)

    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
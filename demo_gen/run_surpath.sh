# CUDA_VISIBLE_DEVICES=3 python main.py \
#     --study tcga_blca \
#     --task survival \
#     --modality survpath \
#     --type_of_path combine \
#     --n_classes 4 \
#     --max_epochs 20 \
#     --lr 0.0005 \
#     --reg 0 \
#     --batch_size 1 \
#     --label_col survival_months_dss \
#     --data_root_dir /data/linzeyu_data/BLCA2/pt_files/ \
#     --label_file datasets_csv/metadata/tcga_blca.csv \
#     --omics_dir datasets_csv/raw_rna_data/combine/blca \
#     --results_dir raw_results \
#     --which_splits 5foldcv \
#     --seed 42 \
#     --wsi_projection_dim 256 \
#     --num_patches 4096 \
#     --bag_loss nll_surv 

# CUDA_VISIBLE_DEVICES=2 python main.py \
#     --study tcga_brca \
#     --task survival \
#     --modality survpath \
#     --type_of_path combine \
#     --n_classes 4 \
#     --max_epochs 20 \
#     --lr 0.0005 \
#     --reg 0 \
#     --alpha_ccd 0.2 \
#     --beta_ccd 0.3 \
#     --batch_size 1 \
#     --label_col survival_months_dss \
#     --data_root_dir /data/TCGA/BRCA/features/pt_files \
#     --label_file datasets_csv/metadata/tcga_brca.csv \
#     --omics_dir datasets_csv/raw_rna_data/combine/brca \
#     --results_dir raw_results \
#     --which_splits 5foldcv \
#     --seed 42 \
#     --wsi_projection_dim 256 \
#     --num_patches 4096 \
#     --use_counterfactual \
#     --bag_loss nll_surv 

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --study tcga_hnsc \
#     --task survival \
#     --modality survpath \
#     --type_of_path combine \
#     --n_classes 4 \
#     --max_epochs 20 \
#     --lr 0.0005 \
#     --reg 0 \
#     --batch_size 1 \
#     --label_col survival_months_dss \
#     --data_root_dir /data/TCGA/HNSC/features/pt_files \
#     --label_file datasets_csv/metadata/tcga_hnsc.csv \
#     --omics_dir datasets_csv/raw_rna_data/combine/hnsc \
#     --results_dir raw_results \
#     --which_splits 5foldcv \
#     --seed 42 \
#     --wsi_projection_dim 256 \
#     --num_patches 4096 \
#     --bag_loss nll_surv 

# CUDA_VISIBLE_DEVICES=2 python main.py \
#     --study tcga_stad \
#     --task survival \
#     --modality survpath \
#     --type_of_path combine \
#     --n_classes 4 \
#     --max_epochs 20 \
#     --lr 0.0005 \
#     --reg 0 \
#     --batch_size 1 \
#     --label_col survival_months_dss \
#     --data_root_dir /data/TCGA/STAD/features/pt_files \
#     --label_file datasets_csv/metadata/tcga_stad.csv \
#     --omics_dir datasets_csv/raw_rna_data/combine/stad \
#     --results_dir raw_results \
#     --which_splits 5foldcv \
#     --seed 42 \
#     --wsi_projection_dim 256 \
#     --num_patches 4096 \
#     --bag_loss nll_surv 


CUDA_VISIBLE_DEVICES=1 python main.py \
    --study tcga_coadread \
    --task survival \
    --modality survpath \
    --type_of_path combine \
    --n_classes 4 \
    --max_epochs 20 \
    --lr 0.0005 \
    --reg 0 \
    --batch_size 1 \
    --label_col survival_months_dss \
    --data_root_dir /data/TCGA/COAD/features/pt_files \
    --label_file datasets_csv/metadata/tcga_coadread.csv \
    --omics_dir datasets_csv/raw_rna_data/combine/coadread \
    --results_dir raw_results \
    --which_splits 5foldcv \
    --seed 42 \
    --wsi_projection_dim 256 \
    --num_patches 4096 \
    --bag_loss nll_surv 
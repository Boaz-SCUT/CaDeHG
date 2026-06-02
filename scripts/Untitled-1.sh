#!/bin/bash

DATA_ROOT_DIR='/data/TCGA/BLCA/features/pt_files' # where are the TCGA features stored?
BASE_DIR="/home/yangzongbao/codes/SurvPath" # where is the repo cloned?
TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="survpath" # what type of model do you want to train?
DIM1=8
DIM2=16
STUDIES=("blca")
LRS=(0.00005 0.0001 0.0005 0.001)
DECAYS=(0.00001 0.0001 0.001 0.01)

for decay in ${DECAYS[@]};
do
    for lr in ${LRS[@]};
    do 
        for STUDY in ${STUDIES[@]};
        do
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --study tcga_blca --task survival --split_dir splits --which_splits 5foldcv \
                --type_of_path combine --modality survpath --data_root_dir /data/TCGA/BLCA/features/pt_files/ --label_file datasets_csv/metadata/tcga_blca.csv \
                --omics_dir datasets_csv/raw_rna_data/combine/blca --results_dir results_blca \
                --batch_size 1 --lr 0.00005 --opt adam --reg 0.00001 \
                --alpha_surv 0.5 --weighted_sample --max_epochs 1 --encoding_dim 1024 \
                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                --encoding_layer_1_dim 8 --encoding_layer_2_dim 16 --encoder_dropout 0.25
        done 
    done
done 
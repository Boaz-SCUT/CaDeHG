from __future__ import print_function, division
from cProfile import label
import os
import pdb
from unittest import case
import pandas as pd
import dgl 
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.general_utils import _series_intersection


ALL_MODALITIES = ['rna_clean.csv']  

class CausalSurvivalDatasetFactory:
    """
    增强的生存数据集工厂，支持临床特征和因果去偏
    """

    def __init__(self,
        study,
        label_file, 
        omics_dir,
        seed, 
        print_info, 
        n_bins, 
        label_col, 
        eps=1e-6,
        num_patches=4096,
        is_mcat=False,
        is_survpath=True,
        type_of_pathway="combine",
        enable_causal=True,  # 新增：启用因果功能
        clinical_features=None,  # 新增：指定使用的临床特征
        ):
        
        # 原有参数
        self.study = study
        self.label_file = label_file
        self.omics_dir = omics_dir
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids  = (None, None)
        self.data_dir = None
        self.label_col = label_col
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.is_mcat = is_mcat
        self.is_survpath = is_survpath
        self.type_of_path = type_of_pathway

        # 新增：因果相关参数
        self.enable_causal = enable_causal
        self.clinical_features = clinical_features or ['age', 'gender', 'stage', 'grade', 'subtype']
        self.clinical_encoders = {}  # 存储分类变量的编码器

        if self.label_col == "survival_months":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"

        #---> process omics data
        self._setup_omics_data() 
        
        #---> labels, metadata, patient_df
        self._setup_metadata_and_labels(eps)

        #---> prepare for weighted sampling
        self._cls_ids_prep()

        #---> load all clinical data 
        self._load_clinical_data()

        #---> 新增：处理临床特征用于因果分析
        if self.enable_causal:
            self._process_clinical_features_for_causal()

        #---> summarize
        self._summarize()

        #---> read the signature files for the correct model/ experiment
        if self.is_mcat:
            self._setup_mcat()
        elif self.is_survpath:
            self._setup_survpath()
        else:
            self.omic_names = []
            self.omic_sizes = []

    def _process_clinical_features_for_causal(self):
        """
        处理临床特征用于因果分析
        包括：数值化分类变量、标准化数值变量、创建环境标签
        """
        print("Processing clinical features for causal analysis...")
        
        # 确保临床数据已加载
        if not hasattr(self, 'clinical_data'):
            print("Warning: Clinical data not loaded, skipping causal feature processing")
            return
        
        # 创建处理后的临床特征DataFrame
        self.processed_clinical_data = self.clinical_data.copy()
        
        # 处理各个临床特征
        for feature in self.clinical_features:
            if feature in self.clinical_data.columns:
                self._process_single_clinical_feature(feature)
            else:
                print(f"Warning: Clinical feature '{feature}' not found in data")
        
        # 创建环境标签（用于IRM训练）
        self._create_environment_labels()
        
        # 标准化数值特征
        self._standardize_clinical_features()
        
        print(f"Processed clinical features: {self.clinical_features}")
        print(f"Clinical data shape: {self.processed_clinical_data.shape}")

    def _process_single_clinical_feature(self, feature):
        """处理单个临床特征"""
        feature_data = self.clinical_data[feature].copy()
        
        # 处理缺失值
        if feature_data.isnull().sum() > 0:
            if feature in ['age']:  # 数值特征用均值填充
                feature_data = feature_data.fillna(feature_data.median())
            else:  # 分类特征用众数填充
                feature_data = feature_data.fillna('Unknown')
        
        # 对分类变量进行编码
        if feature in ['gender', 'stage', 'grade', 'subtype']:
            encoder = LabelEncoder()
            # 处理可能的新类别
            try:
                encoded_data = encoder.fit_transform(feature_data.astype(str))
                self.clinical_encoders[feature] = encoder
                self.processed_clinical_data[feature] = encoded_data
            except Exception as e:
                print(f"Warning: Error encoding {feature}: {e}")
                # 简单的二值化处理
                self.processed_clinical_data[feature] = (feature_data != 'Unknown').astype(int)
        
        elif feature == 'age':
            # 年龄保持数值，但处理异常值
            age_data = pd.to_numeric(feature_data, errors='coerce')
            age_data = age_data.clip(lower=0, upper=100)  # 限制在合理范围
            self.processed_clinical_data[feature] = age_data
        
        else:
            # 其他特征尝试数值化
            try:
                numeric_data = pd.to_numeric(feature_data, errors='coerce')
                if not numeric_data.isnull().all():
                    self.processed_clinical_data[feature] = numeric_data.fillna(0)
                else:
                    # 无法数值化，进行编码
                    encoder = LabelEncoder()
                    encoded_data = encoder.fit_transform(feature_data.astype(str))
                    self.clinical_encoders[feature] = encoder
                    self.processed_clinical_data[feature] = encoded_data
            except:
                self.processed_clinical_data[feature] = 0  # 默认值

    def _create_environment_labels(self):
        """
        创建环境标签用于IRM训练
        基于医院、批次、时间等创建不同的环境
        """
        # 简单的环境划分策略：基于case_id的前缀或随机划分
        case_ids = self.processed_clinical_data.index
        
        # 方法1：基于case_id前缀（TCGA数据通常有组织类型编码）
        if hasattr(self, 'study') and 'tcga' in self.study.lower():
            # TCGA case_id通常格式为 TCGA-XX-XXXX，前面部分表示组织来源
            env_labels = []
            for case_id in case_ids:
                if isinstance(case_id, str) and len(case_id) >= 8:
                    # 使用前8个字符的哈希来创建环境
                    env_id = hash(case_id[:8]) % 3  # 创建3个环境
                else:
                    env_id = 0
                env_labels.append(env_id)
        else:
            # 方法2：随机划分环境
            np.random.seed(self.seed)
            n_envs = 3
            env_labels = np.random.randint(0, n_envs, len(case_ids))
        
        self.processed_clinical_data['environment_id'] = env_labels
        print(f"Created {len(np.unique(env_labels))} environments for IRM training")

    def _standardize_clinical_features(self):
        """标准化临床特征"""
        # 只标准化数值特征
        numeric_features = []
        for feature in self.clinical_features:
            if feature in self.processed_clinical_data.columns:
                if feature == 'age' or self.processed_clinical_data[feature].dtype in ['float64', 'int64']:
                    numeric_features.append(feature)
        
        if numeric_features:
            scaler = StandardScaler()
            self.processed_clinical_data[numeric_features] = scaler.fit_transform(
                self.processed_clinical_data[numeric_features]
            )
            self.clinical_scaler = scaler
            print(f"Standardized features: {numeric_features}")

    def _get_split_from_df(self, args, all_splits, split_key: str='train', fold = None, scaler=None, valid_cols=None):
        """
        修改原有方法以支持临床特征
        """
        if not scaler:
            scaler = {}
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        mask = self.label_data['case_id'].isin(split.tolist())
        df_metadata_slide = args.dataset_factory.label_data.loc[mask, :].reset_index(drop=True)
        
        # select the rna, meth, mut, cnv data for this split
        omics_data_for_split = {}
        for key in args.dataset_factory.all_modalities.keys():
            
            raw_data_df = args.dataset_factory.all_modalities[key]
            mask = raw_data_df.index.isin(split.tolist())
            
            filtered_df = raw_data_df[mask]
            filtered_df = filtered_df[~filtered_df.index.duplicated()] # drop duplicate case_ids
            filtered_df["temp_index"] = filtered_df.index
            filtered_df.reset_index(inplace=True, drop=True)

            clinical_data_mask = self.clinical_data.case_id.isin(split.tolist())
            clinical_data_for_split = self.clinical_data[clinical_data_mask]
            clinical_data_for_split = clinical_data_for_split.set_index("case_id")
            clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")

            # 新增：处理因果相关的临床数据
            if self.enable_causal and hasattr(self, 'processed_clinical_data'):
                processed_clinical_mask = self.processed_clinical_data.index.isin(split.tolist())
                processed_clinical_for_split = self.processed_clinical_data[processed_clinical_mask]
                processed_clinical_for_split = processed_clinical_for_split[~processed_clinical_for_split.index.duplicated(keep='first')]
            else:
                processed_clinical_for_split = None

            # from metadata drop any cases that are not in filtered_df
            mask = [True if item in list(filtered_df["temp_index"]) else False for item in df_metadata_slide.case_id]
            df_metadata_slide = df_metadata_slide[mask]
            df_metadata_slide.reset_index(inplace=True, drop=True)

            mask = [True if item in list(filtered_df["temp_index"]) else False for item in clinical_data_for_split.index]
            clinical_data_for_split = clinical_data_for_split[mask]
            clinical_data_for_split = clinical_data_for_split[~clinical_data_for_split.index.duplicated(keep='first')]
            
            # 对处理后的临床数据应用相同的mask
            if processed_clinical_for_split is not None:
                mask = [True if item in list(filtered_df["temp_index"]) else False for item in processed_clinical_for_split.index]
                processed_clinical_for_split = processed_clinical_for_split[mask]
                processed_clinical_for_split = processed_clinical_for_split[~processed_clinical_for_split.index.duplicated(keep='first')]

            # normalize your df (保持原有的标准化逻辑)
            filtered_normed_df = None
            if split_key in ["val"]:
                
                # store the case_ids -> create a new df without case_ids
                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

                # store original num_patients and num_feats 
                num_patients = df_for_norm.shape[0]
                num_feats = df_for_norm.shape[1]
                columns = {}
                for i in range(num_feats):
                    columns[i] = df_for_norm.columns[i]
                
                # flatten the df into 1D array (make it a column vector)
                flat_df = np.expand_dims(df_for_norm.values.flatten(), 1)

                # get scaler
                scaler_for_data = scaler[key]

                # normalize 
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                # change 1D to 2D
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                # add in case_ids
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

            elif split_key == "train":
                
                # store the case_ids -> create a new df without case_ids
                
                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

                # store original num_patients and num_feats 
                num_patients = df_for_norm.shape[0]
                num_feats = df_for_norm.shape[1]
                columns = {}
                for i in range(num_feats):
                    columns[i] = df_for_norm.columns[i]
                
                # flatten the df into 1D array (make it a column vector)
                flat_df = df_for_norm.values.flatten().reshape(-1, 1)
                
                # get scaler
                scaler_for_data = self._get_scaler(flat_df)

                # normalize 
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                # change 1D to 2D
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                # add in case_ids
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

                # store scaler
                scaler[key] = scaler_for_data
                
            omics_data_for_split[key] = filtered_normed_df

        if split_key == "train":
            sample=True
        elif split_key == "val":
            sample=False
            
        split_dataset = CausalSurvivalDataset(
            split_key=split_key,
            fold=fold,
            study_name=args.study,
            modality=args.modality,
            patient_dict=args.dataset_factory.patient_dict,
            metadata=df_metadata_slide,
            omics_data_dict=omics_data_for_split,
            data_dir=args.data_root_dir,
            num_classes=self.num_classes,
            label_col = self.label_col,
            censorship_var = self.censorship_var,
            valid_cols = valid_cols,
            is_training=split_key=='train',
            clinical_data = clinical_data_for_split,
            processed_clinical_data = processed_clinical_for_split,  # 新增
            enable_causal = self.enable_causal,  # 新增
            num_patches = self.num_patches,
            omic_names = self.omic_names,
            sample=sample
            )

        if split_key == "train":
            return split_dataset, scaler
        else:
            return split_dataset

    # 保持其他原有方法不变
    def _setup_mcat(self):
        self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def _setup_survpath(self):
        self.signatures = pd.read_csv("./datasets_csv/metadata/{}_signatures.csv".format(self.type_of_path))
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def _load_clinical_data(self):
        path_to_data = "./datasets_csv/clinical_data/{}_clinical.csv".format(self.study)
        self.clinical_data = pd.read_csv(path_to_data, index_col=0)
    
    def _setup_omics_data(self):
        self.all_modalities = {}
        for modality in ALL_MODALITIES:
            self.all_modalities[modality.split('_')[0]] = pd.read_csv(
                os.path.join(self.omics_dir, modality),
                engine='python',
                index_col=0
            )

    def _setup_metadata_and_labels(self, eps):
        self.label_data = pd.read_csv(self.label_file, low_memory=False)
        uncensored_df = self._clean_label_data()
        self._discretize_survival_months(eps, uncensored_df)
        self._get_patient_dict()
        self._get_label_dict()
        self._get_patient_data()

    def _clean_label_data(self):
        if "IDC" in self.label_data['oncotree_code']: 
            self.label_data = self.label_data[self.label_data['oncotree_code'] == 'IDC']
        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()
        uncensored_df = self.patients_df[self.patients_df[self.censorship_var] < 1]
        return uncensored_df

    def _discretize_survival_months(self, eps, uncensored_df):
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.label_data[self.label_col].max() + eps
        q_bins[0] = self.label_data[self.label_col].min() - eps
        disc_labels, q_bins = pd.cut(self.patients_df[self.label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patients_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins
        
    def _get_patient_data(self):
        patients_df = self.label_data[~self.label_data.index.duplicated(keep='first')] 
        patient_data = {'case_id': patients_df["case_id"].values, 'label': patients_df['label'].values}
        self.patient_data = patient_data

    def _get_label_dict(self):
        label_dict = {}
        key_count = 0
        for i in range(len(self.bins)-1):
            for c in [0, 1]:
                label_dict.update({(i, c):key_count})
                key_count+=1

        for i in self.label_data.index:
            key = self.label_data.loc[i, 'label']
            self.label_data.at[i, 'disc_label'] = key
            censorship = self.label_data.loc[i, self.censorship_var]
            key = (key, int(censorship))
            self.label_data.at[i, 'label'] = label_dict[key]

        self.num_classes=len(label_dict)
        self.label_dict = label_dict

    def _get_patient_dict(self):
        patient_dict = {}
        temp_label_data = self.label_data.set_index('case_id')
        for patient in self.patients_df['case_id']:
            slide_ids = temp_label_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})
        self.patient_dict = patient_dict
        self.label_data = self.patients_df
        self.label_data.reset_index(drop=True, inplace=True)

    def _cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]   
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.label_data['label'] == i)[0]

    def _summarize(self):
        if self.print_info:
            print("label column: {}".format(self.label_col))
            print("number of cases {}".format(len(self.label_data)))
            print("number of classes: {}".format(self.num_classes))
            if self.enable_causal:
                print("Causal analysis enabled with clinical features: {}".format(self.clinical_features))

    def _get_scaler(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        return scaler
    
    def _apply_scaler(self, data, scaler):
        zero_mask = data == 0
        transformed = scaler.transform(data)
        data = transformed
        data[zero_mask] = 0.
        return data

    def return_splits(self, args, csv_path, fold):
        r"""
        Create the train and val splits for the fold
        
        Args:
            - self
            - args : argspace.Namespace 
            - csv_path : String 
            - fold : Int 
        
        Return: 
            - datasets : tuple 
            
        """

        assert csv_path 
        all_splits = pd.read_csv(csv_path, index_col=0)  # 处理表头格式 ",train,val"
        print("Defining datasets...")
        train_split, scaler = self._get_split_from_df(args, all_splits=all_splits, split_key='train', fold=fold, scaler=None)
        val_split = self._get_split_from_df(args, all_splits=all_splits, split_key='val', fold=fold, scaler=scaler)

        args.omic_sizes = args.dataset_factory.omic_sizes
        datasets = (train_split, val_split)
        
        return datasets

    def _patient_data_prep(self):
        """
        准备患者数据（保持与原版兼容）
        """
        patients = np.unique(np.array(self.label_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.label_data[self.label_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.label_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        """
        静态方法：数据准备（保持与原版兼容）
        """
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        _, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        return len(self.label_data)


class CausalSurvivalDataset(Dataset):
    """
    增强的生存数据集，支持因果去偏
    """

    def __init__(self,
        split_key,
        fold,
        study_name,
        modality,
        patient_dict,
        metadata, 
        omics_data_dict,
        data_dir, 
        num_classes,
        label_col="survival_months_DSS",
        censorship_var = "censorship_DSS",
        valid_cols=None,
        is_training=True,
        clinical_data=-1,
        processed_clinical_data=None,  # 新增
        enable_causal=True,  # 新增
        num_patches=4000,
        omic_names=None,
        sample=True,
        ): 

        super(CausalSurvivalDataset, self).__init__()

        # 原有参数
        self.split_key = split_key
        self.fold = fold
        self.study_name = study_name
        self.modality = modality
        self.patient_dict = patient_dict
        self.metadata = metadata 
        self.omics_data_dict = omics_data_dict
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_training = is_training
        self.clinical_data = clinical_data
        self.num_patches = num_patches
        self.omic_names = omic_names
        self.num_pathways = len(omic_names) if omic_names else 0
        self.sample = sample

        # 新增：因果相关参数
        self.enable_causal = enable_causal
        self.processed_clinical_data = processed_clinical_data

        # for weighted sampling
        self.slide_cls_id_prep()
    
    def slide_cls_id_prep(self):
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.metadata['label'] == i)[0]

    def __getitem__(self, idx):
        """
        修改以支持因果分析所需的额外数据
        """
        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)

        # 获取处理后的临床特征（用于因果分析）
        processed_clinical = None
        environment_id = None
        if self.enable_causal and self.processed_clinical_data is not None:
            try:
                if case_id in self.processed_clinical_data.index:
                    clinical_row = self.processed_clinical_data.loc[case_id]
                    processed_clinical = torch.tensor(clinical_row.drop('environment_id').values, dtype=torch.float32)
                    environment_id = torch.tensor(clinical_row['environment_id'], dtype=torch.long)
                else:
                    # 如果找不到，使用默认值
                    processed_clinical = torch.zeros(len(self.processed_clinical_data.columns) - 1, dtype=torch.float32)
                    environment_id = torch.tensor(0, dtype=torch.long)
            except Exception as e:
                print(f"Warning: Error processing clinical data for {case_id}: {e}")
                processed_clinical = torch.zeros(5, dtype=torch.float32)  # 默认5个特征
                environment_id = torch.tensor(0, dtype=torch.long)

        if self.modality in ['omics', 'snn', 'mlp_per_path']:
            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            
            if self.enable_causal and processed_clinical is not None:
                return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data, processed_clinical, environment_id)
            else:
                return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data)
        
        elif self.modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            
            if self.enable_causal and processed_clinical is not None:
                return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask, processed_clinical, environment_id)
            else:
                return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask)

        elif self.modality in ["coattn", "coattn_motcat"]:
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            omic1 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[0]].iloc[idx])
            omic2 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[1]].iloc[idx])
            omic3 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[2]].iloc[idx])
            omic4 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[3]].iloc[idx])
            omic5 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[4]].iloc[idx])
            omic6 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[5]].iloc[idx])

            if self.enable_causal and processed_clinical is not None:
                return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data, mask, processed_clinical, environment_id)
            else:
                return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data, mask)
        
        elif self.modality == "survpath":
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            omic_list = []
            for i in range(self.num_pathways):
                omic_list.append(torch.tensor(self.omics_data_dict["rna"][self.omic_names[i]].iloc[idx]))
            
            if self.enable_causal and processed_clinical is not None:
                return (patch_features, omic_list, label, event_time, c, clinical_data, mask, processed_clinical, environment_id)
            else:
                return (patch_features, omic_list, label, event_time, c, clinical_data, mask)
        
        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % self.modality)

    # 保持其他原有方法不变
    def get_data_to_return(self, idx):
        case_id = self.metadata['case_id'][idx]
        label = torch.Tensor([self.metadata['disc_label'][idx]])
        event_time = torch.Tensor([self.metadata[self.label_col][idx]])
        c = torch.Tensor([self.metadata[self.censorship_var][idx]])
        slide_ids = self.patient_dict[case_id]
        clinical_data = self.get_clinical_data(case_id)
        return label, event_time, c, slide_ids, clinical_data, case_id
    
    def _load_wsi_embs_from_path(self, data_dir, slide_ids):
        patch_features = []
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi_path)
            patch_features.append(wsi_bag)
        patch_features = torch.cat(patch_features, dim=0)

        if self.sample:
            max_patches = self.num_patches
            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]
        
            if n_samples == max_patches:
                mask = torch.zeros([max_patches])
            else:
                original = patch_features.shape[0]
                how_many_to_add = max_patches - original
                zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                patch_features = torch.concat([patch_features, zeros], dim=0)
                mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])
        else:
            mask = torch.ones([1])

        return patch_features, mask

    def get_clinical_data(self, case_id):
        try:
            stage = self.clinical_data.loc[case_id, "stage"]
        except:
            stage = "N/A"
        try:
            grade = self.clinical_data.loc[case_id, "grade"]
        except:
            grade = "N/A"
        try:
            subtype = self.clinical_data.loc[case_id, "subtype"]
        except:
            subtype = "N/A"
        clinical_data = (stage, grade, subtype)
        return clinical_data
    
    def getlabel(self, idx):
        label = self.metadata['label'][idx]
        return label

    def __len__(self):
        return len(self.metadata)
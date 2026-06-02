# data_explorer.py
import pandas as pd
import argparse

def explore_tcga_data(csv_path):
    """探索TCGA数据结构"""
    
    print("=== TCGA数据结构探索 ===\n")
    
    # 读取数据
    data = pd.read_csv(csv_path, low_memory=False)
    
    print(f"数据集大小: {data.shape}")
    print(f"列数: {len(data.columns)}")
    print("\n=== 所有列名 ===")
    for i, col in enumerate(data.columns):
        print(f"{i+1:2d}. {col}")
    
    print("\n=== 前几行数据 ===")
    print(data.head())
    
    print("\n=== 数据类型 ===")
    print(data.dtypes)
    
    print("\n=== 缺失值统计 ===")
    missing_stats = data.isnull().sum()
    print(missing_stats[missing_stats > 0])
    
    # 寻找可能的临床特征列
    print("\n=== 可能的临床特征列 ===")
    potential_clinical_cols = []
    
    # 年龄相关
    age_keywords = ['age', 'Age', 'AGE']
    age_cols = [col for col in data.columns if any(keyword in col for keyword in age_keywords)]
    if age_cols:
        print(f"年龄相关列: {age_cols}")
        potential_clinical_cols.extend(age_cols)
    
    # 性别相关
    gender_keywords = ['gender', 'sex', 'Gender', 'Sex', 'GENDER', 'SEX']
    gender_cols = [col for col in data.columns if any(keyword in col for keyword in gender_keywords)]
    if gender_cols:
        print(f"性别相关列: {gender_cols}")
        potential_clinical_cols.extend(gender_cols)
    
    # 分期相关
    stage_keywords = ['stage', 'Stage', 'STAGE', 'tumor', 'Tumor', 'TUMOR']
    stage_cols = [col for col in data.columns if any(keyword in col for keyword in stage_keywords)]
    if stage_cols:
        print(f"分期相关列: {stage_cols}")
        potential_clinical_cols.extend(stage_cols)
    
    # 生存相关
    survival_keywords = ['survival', 'time', 'months', 'days', 'death', 'event']
    survival_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in survival_keywords)]
    if survival_cols:
        print(f"生存相关列: {survival_cols}")
        potential_clinical_cols.extend(survival_cols)
    
    # 显示这些列的样本数据
    if potential_clinical_cols:
        print(f"\n=== 潜在临床特征列的样本数据 ===")
        for col in potential_clinical_cols[:10]:  # 只显示前10个
            if col in data.columns:
                print(f"\n{col}:")
                print(f"  类型: {data[col].dtype}")
                print(f"  唯一值数量: {data[col].nunique()}")
                print(f"  唯一值: {data[col].unique()[:10]}")  # 前10个唯一值
                print(f"  值计数: {data[col].value_counts().head()}")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='探索TCGA数据结构')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/tcga_blca_all_clean.csv')
    args = parser.parse_args()
    
    explore_tcga_data(args.csv_path)
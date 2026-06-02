import pandas as pd

def filter_csv_files():
    """
    根据文件2中的case_id筛选文件1中的数据
    """
    try:
        # 读取文件1 - 注意第一列是空的，所以需要跳过
        # 表头：,case_id,stage,grade,subtype
        df1 = pd.read_csv('tcga_brca_clinical.csv', index_col=0)
        print(f"文件1原始数据形状: {df1.shape}")
        print(f"文件1表头: {list(df1.columns)}")
        
        # 读取文件2
        # 表头：case_id,patient_filename,text_report,stage,grade,subtype
        df2 = pd.read_csv('tcga_brca_clinical0.csv')
        print(f"文件2数据形状: {df2.shape}")
        print(f"文件2表头: {list(df2.columns)}")
        
        # 获取文件2中的case_id列表
        case_ids_in_file2 = set(df2['case_id'].unique())
        print(f"文件2中唯一的case_id数量: {len(case_ids_in_file2)}")
        
        # 筛选文件1中的数据，只保留在文件2中存在的case_id
        df1_filtered = df1[df1['case_id'].isin(case_ids_in_file2)]
        print(f"筛选后数据形状: {df1_filtered.shape}")
        
        # 保存筛选后的数据
        df1_filtered.to_csv('tcga_brca_clinical00.csv')
        print(f"筛选后的数据已保存到 'tcga_brca_clinical00.csv'")
        
        # 显示筛选前后的统计信息
        print(f"\n筛选统计:")
        print(f"文件1原始记录数: {len(df1)}")
        print(f"筛选后记录数: {len(df1_filtered)}")
        print(f"保留比例: {len(df1_filtered)/len(df1)*100:.2f}%")
        
        return df1_filtered
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件存在于当前目录:")
        print("- tcga_brca_clinical.csv")
        print("- tcga_brca_clinical0.csv")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    # 运行筛选函数
    filtered_data = filter_csv_files()
    
    # 可选：显示筛选后数据的前几行
    if filtered_data is not None:
        print(f"\n筛选后数据前5行:")
        print(filtered_data.head())
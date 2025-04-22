import pandas as pd
import os

# 读取数据集
def read_dataset(file_path):
    print(f"正在读取 {file_path}...")
    df = pd.read_csv(file_path)
    print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    
    # 查看标签情况
    if 'star' in df.columns:
        print(f"评分分布: \n{df['star'].value_counts()}")
    
    # 输出情感极性（1，-1，0，-2）的分布情况
    aspect_columns = [col for col in df.columns if '#' in col]
    print(f"方面数量: {len(aspect_columns)}")
    
    # 统计每个方面的极性分布
    for col in aspect_columns:
        counts = df[col].value_counts()
        print(f"{col} 极性分布: {dict(counts)}")
    
    # 检查样本数据
    print("前3个样本数据:")
    for i in range(min(3, len(df))):
        print(f"样本 {i+1}:")
        print(f"  ID: {df.iloc[i]['id']}")
        print(f"  评论: {df.iloc[i]['review'][:100]}...")
        print(f"  评分: {df.iloc[i]['star']}")
        for col in aspect_columns:
            print(f"  {col}: {df.iloc[i][col]}")
        print()

# 检查所有数据集
files = ['data/raw/asap_train.csv', 'data/raw/asap_dev.csv', 'data/raw/asap_test.csv']
for file in files:
    try:
        read_dataset(file)
        print("-" * 80)
    except Exception as e:
        print(f"处理 {file} 时出错: {e}")
        print("-" * 80) 
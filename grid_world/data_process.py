import pandas as pd

def update_difficulty_based_on_class2_acc(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 计算 'Class 2 Acc' 列的中位数
    median_value = df['Class 2 Acc'].median()
    print(f"Median of 'Class 2 Acc': {median_value}")
    
    # 创建一个新的 DataFrame，按 'Class 2 Acc' 列排序
    sorted_df = df.sort_values(by='Class 2 Acc').reset_index()
    
    # 计算中间索引
    mid_index = len(sorted_df) // 2
    
    # 前 50% 赋值为 'easy'
    sorted_df.loc[:mid_index, 'difficulty'] = 'easy'
    
    # 后 50% 赋值为 'difficult'
    sorted_df.loc[mid_index:, 'difficulty'] = 'difficult'
    
    # 将更新后的 'difficulty' 列赋值回原始 DataFrame
    df['difficulty'] = sorted_df.set_index('index')['difficulty']
    
    # 保存更新后的数据到 CSV 文件
    df.to_csv(csv_path, index=False)
    print(f"Updated difficulty based on 'Class 2 Acc' and saved to {csv_path}")

# 使用函数更新数据
csv_path = 'fire_data.csv'
update_difficulty_based_on_class2_acc(csv_path)

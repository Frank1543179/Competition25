import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('..//CC07_1_13_shuffled.csv')

# 按年龄分组（5岁为步长）
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50',
          '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100',
          '100-105', '105-110', '110-115', '115-120']
df['age_group'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

# 显示分组结果
print("年龄分组统计：")
print(df['age_group'].value_counts().sort_index())
print("\n")

# 在每个年龄组内打乱数据（保持年龄不变）
shuffled_dfs = []

for group_name, group_df in df.groupby('age_group'):
    # 复制组内数据
    shuffled_group = group_df.copy()

    # 获取除AGE外的所有列
    other_columns = [col for col in shuffled_group.columns if col != 'AGE' and col != 'age_group']

    # 对这些列进行行间打乱
    for col in other_columns:
        shuffled_group[col] = np.random.permutation(shuffled_group[col].values)

    shuffled_dfs.append(shuffled_group)

# 合并所有打乱后的数据
final_df = pd.concat(shuffled_dfs, ignore_index=True)

# 显示前几行打乱后的数据
print("打乱后的数据前10行：")
print(final_df.head(10))
print("\n")

# 验证年龄分组是否保持不变
print("打乱后各年龄组统计：")
print(final_df['age_group'].value_counts().sort_index())
print("\n")

# 验证年龄值是否保持不变
print("原始数据年龄范围：", df['AGE'].min(), "-", df['AGE'].max())
print("打乱后年龄范围：", final_df['AGE'].min(), "-", final_df['AGE'].max())
print("年龄值是否相同：", set(df['AGE']) == set(final_df['AGE']))

# 删除age_group列，使列数与原始数据相同
final_df_cleaned = final_df.drop('age_group', axis=1)

# 验证列数是否与原始数据相同
print(f"原始数据列数: {len(df.columns)}")
print(f"最终数据列数: {len(final_df_cleaned.columns)}")
print(f"列数是否相同: {len(df.columns) == len(final_df_cleaned.columns)}")

# 保存为CSV文件（不包含age_group列）
final_df_cleaned.to_csv('BB07_1_group1.csv', index=False)
print("\n数据已保存为 'BB07_1_group1.csv'")

# 显示各年龄组的样本数量
print("\n各年龄组详细统计：")
age_group_stats = df.groupby('age_group').size()
for group, count in age_group_stats.items():
    age_range = f"{group}"
    print(f"{age_range}岁: {count}个样本")
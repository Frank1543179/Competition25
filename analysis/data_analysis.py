import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置图形样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'BB07_1.csv')

# 读取数据
df = pd.read_csv(data_path)

print("数据基本信息:")
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 1. GENDER列分析 - 各性别占比
print("\n" + "=" * 50)
print("GENDER 分析")
print("=" * 50)

gender_counts = df['GENDER'].value_counts()
gender_percent = df['GENDER'].value_counts(normalize=True) * 100

print(f"男性: {gender_counts.get('M', 0)}人 ({gender_percent.get('M', 0):.2f}%)")
print(f"女性: {gender_counts.get('F', 0)}人 ({gender_percent.get('F', 0):.2f}%)")

# 可视化
plt.figure(figsize=(8, 6))
plt.bar(gender_counts.index, gender_percent.values, color=['lightblue', 'lightpink'])
plt.ylabel('Proportion (%)')
plt.title('Gender distribution')
for i, (count, percent) in enumerate(zip(gender_counts.values, gender_percent.values)):
    plt.text(i, percent + 1, f'{count}人\n({percent:.1f}%)', ha='center', va='bottom')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# 2. AGE列分析 - 按年龄段分组
print("\n" + "=" * 50)
print("AGE 分析")
print("=" * 50)


# 创建年龄段
def create_age_group(age):
    if age <= 10:
        return '0-10'
    elif age <= 20:
        return '11-20'
    elif age <= 30:
        return '21-30'
    elif age <= 40:
        return '31-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    elif age <= 70:
        return '61-70'
    elif age <= 80:
        return '71-80'
    elif age <= 90:
        return '81-90'
    elif age <= 100:
        return '91-100'
    else:
        return '100+'


df['age_group'] = df['AGE'].apply(create_age_group)
age_group_counts = df['age_group'].value_counts().sort_index()

print(f"年龄最小值: {df['AGE'].min()}")
print(f"年龄最大值: {df['AGE'].max()}")
print(f"年龄平均值: {df['AGE'].mean():.2f}")
print(f"\n各年龄段人数:")
for group, count in age_group_counts.items():
    print(f"{group}岁: {count}人")

# 可视化
plt.figure(figsize=(12, 6))
plt.bar(age_group_counts.index, age_group_counts.values, color='skyblue')
plt.ylabel('Number of people')
plt.title('Age Group Distribution')
plt.xticks(rotation=45)
for i, count in enumerate(age_group_counts.values):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 3. RACE列分析
print("\n" + "=" * 50)
print("RACE 分析")
print("=" * 50)

race_counts = df['RACE'].value_counts()
race_percent = df['RACE'].value_counts(normalize=True) * 100

print("各种族人数:")
for race, count in race_counts.items():
    print(f"{race}: {count}人 ({race_percent[race]:.2f}%)")

# 创建others种族（合并native、other、hawaiian）
df['race_grouped'] = df['RACE'].apply(lambda x: 'others' if x in ['native', 'other', 'hawaiian'] else x)
race_grouped_counts = df['race_grouped'].value_counts()
race_grouped_percent = df['race_grouped'].value_counts(normalize=True) * 100

print(f"\n合并后各种族人数:")
for race, count in race_grouped_counts.items():
    print(f"{race}: {count}人 ({race_grouped_percent[race]:.2f}%)")

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 原始种族分布
ax1.bar(race_counts.index, race_percent.values, color='lightgreen')
ax1.set_ylabel('Proportion (%)')
ax1.set_title('Original ethnic distribution')
ax1.tick_params(axis='x', rotation=45)
for i, (count, percent) in enumerate(zip(race_counts.values, race_percent.values)):
    ax1.text(i, percent + 1, f'{count}人\n({percent:.1f}%)', ha='center', va='bottom', fontsize=8)

# 合并后种族分布
ax2.bar(race_grouped_counts.index, race_grouped_percent.values, color='lightcoral')
ax2.set_ylabel('Proportion (%)')
ax2.set_title('Racial distribution after merging (native+other+hawaiian=others)')
ax2.tick_params(axis='x', rotation=45)
for i, (count, percent) in enumerate(zip(race_grouped_counts.values, race_grouped_percent.values)):
    ax2.text(i, percent + 1, f'{count}人\n({percent:.1f}%)', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 4. ETHNICITY列分析
print("\n" + "=" * 50)
print("ETHNICITY 分析")
print("=" * 50)

ethnicity_counts = df['ETHNICITY'].value_counts()
ethnicity_percent = df['ETHNICITY'].value_counts(normalize=True) * 100

print("各民族人数:")
for eth, count in ethnicity_counts.items():
    print(f"{eth}: {count}人 ({ethnicity_percent[eth]:.2f}%)")

# 可视化
plt.figure(figsize=(8, 6))
plt.bar(ethnicity_counts.index, ethnicity_percent.values, color='gold')
plt.ylabel('Proportion (%)')
plt.title('Ethnic distribution')
plt.xticks(rotation=45)
for i, (count, percent) in enumerate(zip(ethnicity_counts.values, ethnicity_percent.values)):
    plt.text(i, percent + 1, f'{count}人\n({percent:.1f}%)', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 5. 医疗计数指标分析
print("\n" + "=" * 50)
print("医疗计数指标分析")
print("=" * 50)

medical_columns = ['encounter_count', 'num_procedures', 'num_medications',
                   'num_immunizations', 'num_allergies', 'num_devices']

# 按年龄段分组计算平均值
medical_by_age = df.groupby('age_group')[medical_columns].mean()

print("各医疗计数指标统计:")
for col in medical_columns:
    print(f"\n{col}:")
    print(f"  最小值: {df[col].min()}")
    print(f"  最大值: {df[col].max()}")
    print(f"  平均值: {df[col].mean():.2f}")

# 可视化
plt.figure(figsize=(14, 8))
x = np.arange(len(medical_by_age.index))
width = 0.12
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for i, col in enumerate(medical_columns):
    plt.bar(x + i * width, medical_by_age[col], width, label=col, color=colors[i])

plt.xlabel('Age group')
plt.ylabel('Average number of times')
plt.title('Average values of medical technical indicators for all age groups')
plt.xticks(x + width * 2.5, medical_by_age.index, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6. 疾病标志位分析
print("\n" + "=" * 50)
print("疾病标志位分析")
print("=" * 50)

flag_columns = ['asthma_flag', 'stroke_flag', 'obesity_flag', 'depression_flag']

# 按年龄段分组计算平均值（即患病率）
flags_by_age = df.groupby('age_group')[flag_columns].mean()

print("各疾病标志位统计:")
for col in flag_columns:
    total_cases = df[col].sum()
    prevalence = total_cases / len(df) * 100
    print(f"\n{col}:")
    print(f"  总病例数: {total_cases}")
    print(f"  总患病率: {prevalence:.2f}%")

# 可视化
plt.figure(figsize=(14, 8))
x = np.arange(len(flags_by_age.index))
width = 0.2
colors = ['red', 'blue', 'green', 'orange']

for i, col in enumerate(flag_columns):
    plt.bar(x + i * width, flags_by_age[col] * 100, width, label=col, color=colors[i])

plt.xlabel('Age group')
plt.ylabel('Prevalence (%)')
plt.title('Disease prevalence across all age groups')
plt.xticks(x + width * 1.5, flags_by_age.index, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 7. 生理指标分析（修复版）
print("\n" + "=" * 50)
print("生理指标分析")
print("=" * 50)

# 定义正常范围（根据医学标准）
normal_ranges = {
    'mean_systolic_bp': (90, 140),  # 收缩压正常范围
    'mean_diastolic_bp': (60, 90),  # 舒张压正常范围
    'mean_bmi': (18.5, 24.9),  # BMI正常范围
    'mean_weight': (50, 100)  # 体重正常范围（kg）
}

physio_columns = ['mean_systolic_bp', 'mean_diastolic_bp', 'mean_bmi', 'mean_weight']

# 计算各年龄段正常和异常人数
for col in physio_columns:
    low, high = normal_ranges[col]
    df[f'{col}_normal'] = (df[col] >= low) & (df[col] <= high)

# 修复：可视化每个生理指标
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(physio_columns):
    # 按年龄段和正常状态分组计数
    physio_by_age = df.groupby(['age_group', f'{col}_normal']).size().unstack(fill_value=0)

    # 确保所有年龄段都有数据
    all_age_groups = sorted(df['age_group'].unique())
    physio_by_age = physio_by_age.reindex(all_age_groups, fill_value=0)

    # 确保有正常(True)和异常(False)两列
    for status in [True, False]:
        if status not in physio_by_age.columns:
            physio_by_age[status] = 0

    # 绘制堆叠柱状图 - 修复：使用正确的列名
    bottom = np.zeros(len(physio_by_age))

    # 正常数据（绿色）
    axes[i].bar(physio_by_age.index, physio_by_age[True], bottom=bottom,
                color='green', label='正常', alpha=0.7)
    bottom += physio_by_age[True]

    # 异常数据（红色）
    axes[i].bar(physio_by_age.index, physio_by_age[False], bottom=bottom,
                color='red', label='异常', alpha=0.7)

    # 设置图表标题和标签
    col_names = {
        'mean_systolic_bp': 'Systolic blood pressure',
        'mean_diastolic_bp': 'Diastolic blood pressure',
        'mean_bmi': 'BMI',
        'mean_weight': 'Weight'
    }

    axes[i].set_title(f'{col_names[col]} Normal/Abnormal')
    axes[i].set_xlabel('Age group')
    axes[i].set_ylabel('Number of people')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()

    # 添加数值标签 - 修复：使用正确的列名
    for idx, age_group in enumerate(physio_by_age.index):
        total = physio_by_age.loc[age_group].sum()
        normal_count = physio_by_age.loc[age_group, True]  # 修复：使用True而不是'正常'
        if total > 0:
            normal_percent = (normal_count / total * 100)
            axes[i].text(idx, total + 0.5, f'{normal_percent:.1f}%',
                         ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 输出各生理指标统计
print("\n生理指标统计:")
for col in physio_columns:
    low, high = normal_ranges[col]
    normal_count = df[f'{col}_normal'].sum()
    abnormal_count = len(df) - normal_count
    normal_percent = normal_count / len(df) * 100

    col_names = {
        'mean_systolic_bp': '收缩压',
        'mean_diastolic_bp': '舒张压',
        'mean_bmi': 'BMI指数',
        'mean_weight': '体重'
    }

    print(f"\n{col_names[col]}:")
    print(f"  正常范围: {low} - {high}")
    print(f"  正常人数: {normal_count} ({normal_percent:.2f}%)")
    print(f"  异常人数: {abnormal_count} ({100 - normal_percent:.2f}%)")
    print(f"  最小值: {df[col].min():.2f}")
    print(f"  最大值: {df[col].max():.2f}")
    print(f"  平均值: {df[col].mean():.2f}")
    print(f"  标准差: {df[col].std():.2f}")
    print(f"  中位数: {df[col].median():.2f}")

# 8. 新增：更深入的数据特征分析（便于匿名化和有用性评估）
print("\n" + "=" * 50)
print("深入数据特征分析（匿名化和有用性评估）")
print("=" * 50)

# 8.1 数值特征的分布分析
print("\n1. 数值特征的分布特征:")
numeric_features = ['AGE', 'encounter_count', 'num_procedures', 'num_medications',
                    'num_immunizations', 'num_allergies', 'num_devices'] + physio_columns

for feature in numeric_features:
    print(f"\n{feature}:")
    print(f"  偏度: {df[feature].skew():.3f} (>{1}或<{-1}表示严重偏斜)")
    print(f"  峰度: {df[feature].kurtosis():.3f} (正态分布≈0)")
    print(f"  变异系数: {(df[feature].std() / df[feature].mean()):.3f} (值越大变异越大)")

# 8.2 相关性分析
print("\n2. 主要数值特征的相关性矩阵:")
correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('数值特征相关性热图')
plt.tight_layout()
plt.show()

# 8.3 异常值检测
print("\n3. 异常值分析（使用IQR方法）:")
for feature in numeric_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"{feature}: {len(outliers)}个异常值 ({len(outliers) / len(df) * 100:.2f}%)")

# 8.4 数据稀疏性分析
print("\n4. 数据稀疏性分析:")
sparse_features = ['num_allergies', 'num_devices', 'asthma_flag', 'stroke_flag',
                   'obesity_flag', 'depression_flag']

for feature in sparse_features:
    zero_count = (df[feature] == 0).sum()
    zero_percent = zero_count / len(df) * 100
    print(f"{feature}: {zero_count}个零值 ({zero_percent:.2f}%)")

# 8.5 唯一值分析（重识别风险）
print("\n5. 唯一值分析（重识别风险评估）:")
for feature in ['AGE', 'RACE', 'ETHNICITY', 'GENDER']:
    unique_count = df[feature].nunique()
    uniqueness_ratio = unique_count / len(df)
    print(f"{feature}: {unique_count}个唯一值, 唯一性比率: {uniqueness_ratio:.4f}")

# 8.6 组合唯一性分析（准标识符分析）
print("\n6. 准标识符组合唯一性:")
quasi_identifiers = ['AGE', 'GENDER', 'RACE']
combination = df[quasi_identifiers].astype(str).apply(lambda x: '_'.join(x), axis=1)
unique_combinations = combination.nunique()
print(f"组合 {quasi_identifiers} 的唯一组合数: {unique_combinations}")
print(f"重识别风险: {unique_combinations / len(df) * 100:.2f}%")

# 8.7 数据效用评估
print("\n7. 数据效用评估:")
# 信息熵（分类变量）
from scipy.stats import entropy

for feature in ['RACE', 'ETHNICITY', 'GENDER']:
    value_counts = df[feature].value_counts(normalize=True)
    feature_entropy = entropy(value_counts)
    print(f"{feature} 信息熵: {feature_entropy:.3f} (越高表示信息量越大)")

# 8.8 可视化：各年龄段生理指标分布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(physio_columns):
    # 使用小提琴图展示分布
    sns.violinplot(data=df, x='age_group', y=col, ax=axes[i])
    axes[i].set_title(f'{col_names[col]} 各年龄段分布')
    axes[i].set_xlabel('年龄段')
    axes[i].set_ylabel(col_names[col])
    axes[i].tick_params(axis='x', rotation=45)

    # 添加正常范围线
    low, high = normal_ranges[col]
    axes[i].axhline(y=low, color='red', linestyle='--', alpha=0.7, label=f'正常下限: {low}')
    axes[i].axhline(y=high, color='red', linestyle='--', alpha=0.7, label=f'正常上限: {high}')
    axes[i].legend()

plt.tight_layout()
plt.show()

print("\n分析完成!")
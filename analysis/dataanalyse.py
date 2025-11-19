import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import entropy

# 设置图形样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'BB07_1.csv')

# 读取数据
df = pd.read_csv(data_path)

print("Basic Data Information:")
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 1. GENDER列分析 - 各性别占比
print("\n" + "=" * 50)
print("GENDER Analysis")
print("=" * 50)

gender_counts = df['GENDER'].value_counts()
gender_percent = df['GENDER'].value_counts(normalize=True) * 100

print(f"Male: {gender_counts.get('M', 0)} people ({gender_percent.get('M', 0):.2f}%)")
print(f"Female: {gender_counts.get('F', 0)} people ({gender_percent.get('F', 0):.2f}%)")

# 可视化
plt.figure(figsize=(8, 6))
plt.bar(gender_counts.index, gender_percent.values, color=['lightblue', 'lightpink'])
plt.ylabel('Proportion (%)')
plt.title('Gender Distribution')
for i, (count, percent) in enumerate(zip(gender_counts.values, gender_percent.values)):
    plt.text(i, percent + 1, f'{count} people\n({percent:.1f}%)', ha='center', va='bottom')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# 2. AGE列分析 - 按年龄段分组
print("\n" + "=" * 50)
print("AGE Analysis")
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

print(f"Min Age: {df['AGE'].min()}")
print(f"Max Age: {df['AGE'].max()}")
print(f"Mean Age: {df['AGE'].mean():.2f}")
print(f"\nAge Group Distribution:")
for group, count in age_group_counts.items():
    print(f"{group} years: {count} people")

# 可视化
plt.figure(figsize=(12, 6))
plt.bar(age_group_counts.index, age_group_counts.values, color='skyblue')
plt.ylabel('Number of People')
plt.title('Age Group Distribution')
plt.xticks(rotation=45)
for i, count in enumerate(age_group_counts.values):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 3. RACE列分析
print("\n" + "=" * 50)
print("RACE Analysis")
print("=" * 50)

race_counts = df['RACE'].value_counts()
race_percent = df['RACE'].value_counts(normalize=True) * 100

print("Race Distribution:")
for race, count in race_counts.items():
    print(f"{race}: {count} people ({race_percent[race]:.2f}%)")

# 创建others种族（合并native、other、hawaiian）
df['race_grouped'] = df['RACE'].apply(lambda x: 'others' if x in ['native', 'other', 'hawaiian'] else x)
race_grouped_counts = df['race_grouped'].value_counts()
race_grouped_percent = df['race_grouped'].value_counts(normalize=True) * 100

print(f"\nRace Distribution (Grouped):")
for race, count in race_grouped_counts.items():
    print(f"{race}: {count} people ({race_grouped_percent[race]:.2f}%)")

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 原始种族分布
ax1.bar(race_counts.index, race_percent.values, color='lightgreen')
ax1.set_ylabel('Proportion (%)')
ax1.set_title('Original Race Distribution')
ax1.tick_params(axis='x', rotation=45)
for i, (count, percent) in enumerate(zip(race_counts.values, race_percent.values)):
    ax1.text(i, percent + 1, f'{count}\n({percent:.1f}%)', ha='center', va='bottom', fontsize=8)

# 合并后种族分布
ax2.bar(race_grouped_counts.index, race_grouped_percent.values, color='lightcoral')
ax2.set_ylabel('Proportion (%)')
ax2.set_title('Race Distribution (Grouped)')
ax2.tick_params(axis='x', rotation=45)
for i, (count, percent) in enumerate(zip(race_grouped_counts.values, race_grouped_percent.values)):
    ax2.text(i, percent + 1, f'{count}\n({percent:.1f}%)', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 4. ETHNICITY列分析
print("\n" + "=" * 50)
print("ETHNICITY Analysis")
print("=" * 50)

ethnicity_counts = df['ETHNICITY'].value_counts()
ethnicity_percent = df['ETHNICITY'].value_counts(normalize=True) * 100

print("Ethnicity 分析:")
for eth, count in ethnicity_counts.items():
    print(f"{eth}: {count} people ({ethnicity_percent[eth]:.2f}%)")

# 可视化
plt.figure(figsize=(8, 6))
plt.bar(ethnicity_counts.index, ethnicity_percent.values, color='gold')
plt.ylabel('Proportion (%)')
plt.title('Ethnicity Distribution')
plt.xticks(rotation=45)
for i, (count, percent) in enumerate(zip(ethnicity_counts.values, ethnicity_percent.values)):
    plt.text(i, percent + 1, f'{count}\n({percent:.1f}%)', ha='center', va='bottom')
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

plt.xlabel('Age Group')
plt.ylabel('Average Count')
plt.title('Average Medical Counts by Age Group')
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
    print(f"  患病率: {prevalence:.2f}%")

# 可视化
plt.figure(figsize=(14, 8))
x = np.arange(len(flags_by_age.index))
width = 0.2
colors = ['red', 'blue', 'green', 'orange']

for i, col in enumerate(flag_columns):
    plt.bar(x + i * width, flags_by_age[col] * 100, width, label=col, color=colors[i])

plt.xlabel('Age Group')
plt.ylabel('Prevalence (%)')
plt.title('Disease Prevalence by Age Group')
plt.xticks(x + width * 1.5, flags_by_age.index, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 7. 生理指标分析（修正版）
print("\n" + "=" * 50)
print("Physiological Indicators Analysis")
print("=" * 50)

physio_columns = ['mean_systolic_bp', 'mean_diastolic_bp', 'mean_bmi', 'mean_weight']

# 修正的正常范围定义
def get_normal_range(column, age):
    """根据年龄和指标类型返回适当的正常范围"""
    if column == 'mean_bmi':
        # BMI正常范围（针对成年人）
        if age >= 18:
            return (18.5, 24.9)  # 成年人正常BMI
        else:
            # 儿童BMI需要年龄特异性标准，这里使用更宽松的范围
            return (15.0, 30.0)  # 儿童临时范围

    elif column == 'mean_weight':
        # 体重需要年龄和性别特异性标准
        # 这里使用简化的年龄分组
        if age < 12:
            return (15, 60)  # 儿童
        elif age < 18:
            return (40, 80)  # 青少年
        else:
            return (45, 100)  # 成年人

    elif column == 'mean_systolic_bp':
        # 血压正常范围
        if age < 18:
            return (80, 120)  # 儿童青少年
        else:
            return (90, 140)  # 成年人

    elif column == 'mean_diastolic_bp':
        # 舒张压
        if age < 18:
            return (50, 80)  # 儿童青少年
        else:
            return (60, 90)  # 成年人

    return None


# 计算各指标的正常状态（基于年龄调整）
for col in physio_columns:
    normal_flags = []
    for idx, row in df.iterrows():
        age = row['AGE']
        value = row[col]
        normal_range = get_normal_range(col, age)

        if normal_range:
            low, high = normal_range
            is_normal = (value >= low) & (value <= high)
        else:
            is_normal = True  # 如果没有定义范围，默认正常

        normal_flags.append(is_normal)

    df[f'{col}_normal'] = normal_flags

# 可视化每个生理指标
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

    # 绘制堆叠柱状图
    bottom = np.zeros(len(physio_by_age))

    # 正常数据（绿色）
    axes[i].bar(physio_by_age.index, physio_by_age[True], bottom=bottom,
                color='green', label='Normal', alpha=0.7)
    bottom += physio_by_age[True]

    # 异常数据（红色）
    axes[i].bar(physio_by_age.index, physio_by_age[False], bottom=bottom,
                color='red', label='Abnormal', alpha=0.7)

    # 设置图表标题和标签
    col_names = {
        'mean_systolic_bp': 'Systolic BP',
        'mean_diastolic_bp': 'Diastolic BP',
        'mean_bmi': 'BMI',
        'mean_weight': 'Weight'
    }

    axes[i].set_title(f'{col_names[col]} - Normal/Abnormal Distribution')
    axes[i].set_xlabel('Age Group')
    axes[i].set_ylabel('Number of People')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()

    # 添加数值标签
    for idx, age_group in enumerate(physio_by_age.index):
        total = physio_by_age.loc[age_group].sum()
        normal_count = physio_by_age.loc[age_group, True]
        if total > 0:
            normal_percent = (normal_count / total * 100)
            axes[i].text(idx, total + 0.5, f'{normal_percent:.1f}%',
                         ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 输出各生理指标统计
print("\nPhysiological Indicators Statistics (Age-Adjusted Ranges):")
for col in physio_columns:
    # 计算总体统计
    normal_count = df[f'{col}_normal'].sum()
    abnormal_count = len(df) - normal_count
    normal_percent = normal_count / len(df) * 100

    col_names = {
        'mean_systolic_bp': 'Systolic BP',
        'mean_diastolic_bp': 'Diastolic BP',
        'mean_bmi': 'BMI',
        'mean_weight': 'Weight'
    }

    print(f"\n{col_names[col]}:")
    print(f"  Normal: {normal_count} ({normal_percent:.2f}%)")
    print(f"  Abnormal: {abnormal_count} ({100 - normal_percent:.2f}%)")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Std: {df[col].std():.2f}")
    print(f"  Median: {df[col].median():.2f}")

    # 按年龄段显示正常率
    print(f"  Normal rates by age group:")
    for age_group in sorted(df['age_group'].unique()):
        group_data = df[df['age_group'] == age_group]
        if len(group_data) > 0:
            group_normal = group_data[f'{col}_normal'].sum()
            group_normal_pct = group_normal / len(group_data) * 100
            print(f"    {age_group}: {group_normal_pct:.1f}%")

# 额外的BMI分析
print("\n" + "=" * 50)
print("Detailed BMI Analysis")
print("=" * 50)


# BMI分类（WHO标准）
def classify_bmi(bmi, age):
    if age < 18:
        return "Child/Adolescent"  # 需要专门的儿童BMI标准
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


df['bmi_category'] = df.apply(lambda row: classify_bmi(row['mean_bmi'], row['AGE']), axis=1)
bmi_counts = df['bmi_category'].value_counts()

print("BMI Categories (Adults 18+ only):")
adults = df[df['AGE'] >= 18]
adult_bmi_counts = adults['bmi_category'].value_counts()
for category, count in adult_bmi_counts.items():
    percent = count / len(adults) * 100
    print(f"  {category}: {count} ({percent:.1f}%)")

print(f"\nChildren/Adolescents (<18): {len(df[df['AGE'] < 18])} people")

# BMI分布可视化
plt.figure(figsize=(12, 8))

# 成年人BMI分布
adult_bmi = df[df['AGE'] >= 18]['mean_bmi']
plt.subplot(2, 2, 1)
plt.hist(adult_bmi, bins=30, alpha=0.7, color='blue')
plt.axvline(x=18.5, color='red', linestyle='--', label='Underweight')
plt.axvline(x=25, color='green', linestyle='--', label='Normal')
plt.axvline(x=30, color='orange', linestyle='--', label='Overweight')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Adult BMI Distribution (18+)')
plt.legend()

# 儿童BMI分布
child_bmi = df[df['AGE'] < 18]['mean_bmi']
plt.subplot(2, 2, 2)
plt.hist(child_bmi, bins=30, alpha=0.7, color='green')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Child/Adolescent BMI Distribution (<18)')

# 按年龄组的BMI趋势
plt.subplot(2, 2, 3)
bmi_by_age = df.groupby('age_group')['mean_bmi'].mean()
plt.bar(bmi_by_age.index, bmi_by_age.values, color='purple', alpha=0.7)
plt.xlabel('Age Group')
plt.ylabel('Average BMI')
plt.title('Average BMI by Age Group')
plt.xticks(rotation=45)

# 体重分布
plt.subplot(2, 2, 4)
plt.hist(df['mean_weight'], bins=30, alpha=0.7, color='brown')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.title('Weight Distribution')

plt.tight_layout()
plt.show()

# 8. 深入数据特征分析
print("\n" + "=" * 50)
print("高级数据分析（匿名化与实用性）")
print("=" * 50)

# 8.1 数值特征的分布分析
print("\n1. 分布特征:")
numeric_features = ['AGE', 'encounter_count', 'num_procedures', 'num_medications',
                    'num_immunizations', 'num_allergies', 'num_devices'] + physio_columns

for feature in numeric_features:
    print(f"\n{feature}:")
    print(f"  Skewness: {df[feature].skew():.3f} (>1 or <-1 indicates high skew)")
    print(f"  Kurtosis: {df[feature].kurtosis():.3f} (normal ≈ 0)")
    print(f"  CV: {(df[feature].std() / df[feature].mean()):.3f} (higher = more variation)")

# 8.2 相关性分析
print("\n2. 相关矩阵:")
correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix - Numerical Features')
plt.tight_layout()
plt.show()

# 8.3 异常值检测
print("\n3. 异常值分析（IQR 方法）:")
for feature in numeric_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"{feature}: {len(outliers)} outliers ({len(outliers) / len(df) * 100:.2f}%)")

# 8.4 数据稀疏性分析
print("\n4. 数据稀疏性分析：")
sparse_features = ['num_allergies', 'num_devices', 'asthma_flag', 'stroke_flag',
                   'obesity_flag', 'depression_flag']

for feature in sparse_features:
    zero_count = (df[feature] == 0).sum()
    zero_percent = zero_count / len(df) * 100
    print(f"{feature}: {zero_count} zeros ({zero_percent:.2f}%)")

# 8.5 唯一值分析
print("\n5. 唯一性分析（重新识别风险）::")
for feature in ['AGE', 'RACE', 'ETHNICITY', 'GENDER']:
    unique_count = df[feature].nunique()
    uniqueness_ratio = unique_count / len(df)
    print(f"{feature}: {unique_count} unique values, uniqueness ratio: {uniqueness_ratio:.4f}")

# 8.6 组合唯一性分析
print("\n6. 准一标识符组合分析:")
quasi_identifiers = ['AGE', 'GENDER', 'RACE']
combination = df[quasi_identifiers].astype(str).apply(lambda x: '_'.join(x), axis=1)
unique_combinations = combination.nunique()
print(f"组合 {quasi_identifiers}: {unique_combinations} 独特组合")
print(f"重新识别风险： {unique_combinations / len(df) * 100:.2f}%")

# 8.7 数据效用评估
print("\n7. Data Utility Assessment:")
for feature in ['RACE', 'ETHNICITY', 'GENDER']:
    value_counts = df[feature].value_counts(normalize=True)
    feature_entropy = entropy(value_counts)
    print(f"{feature} entropy: {feature_entropy:.3f} (higher = more information)")

# 8.8 可视化：各年龄段生理指标分布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(physio_columns):
    # 使用小提琴图展示分布
    sns.violinplot(data=df, x='age_group', y=col, ax=axes[i])
    col_names = {
        'mean_systolic_bp': 'Systolic BP',
        'mean_diastolic_bp': 'Diastolic BP',
        'mean_bmi': 'BMI',
        'mean_weight': 'Weight'
    }
    axes[i].set_title(f'{col_names[col]} Distribution by Age Group')
    axes[i].set_xlabel('Age Group')
    axes[i].set_ylabel(col_names[col])
    axes[i].tick_params(axis='x', rotation=45)

    # 定义正常范围（根据医学标准）
    normal_ranges = {
        'mean_systolic_bp': (90, 140),  # 收缩压正常范围
        'mean_diastolic_bp': (60, 90),  # 舒张压正常范围
        'mean_bmi': (18.5, 24.9),  # BMI正常范围
        'mean_weight': (50, 100)  # 体重正常范围（kg）
    }

    # 添加正常范围线
    low, high = normal_ranges[col]
    axes[i].axhline(y=low, color='red', linestyle='--', alpha=0.7, label=f'Lower: {low}')
    axes[i].axhline(y=high, color='red', linestyle='--', alpha=0.7, label=f'Upper: {high}')
    axes[i].legend()

plt.tight_layout()
plt.show()

# 9. 匿名化策略建议
print("\n" + "=" * 50)
print("匿名化策略建议")
print("=" * 50)

print("\n推荐的匿名化技术:")
print("1. AGE:概括为10年一个区间（已实现）")
print("2. RACE: 将稀有类别分组（本地其他夏威夷 → 其他）")
print("3. ETHNICITY: 保持原样（仅两个类别）")
print("4. 敏感医疗数据：添加适度噪声（±10-20%）")
print("5.生理测量：四舍五入到临床相关的精度")
print("6.确保准标识符的 k-匿名性，其中 k ≥ 5")

print("\n期望有用性：")
print("- 通过分组保留的年龄模式")
print("- 疾病流行模式保持")
print("- 医疗趋势相关性已保留")
print("- 统计分布基本保持不变")

print("\n隐私增强")
print("- 将重新识别风险从100%降低至<20%")
print("- 保护敏感极值")
print("- 保持人口统计多样性")

print("\n分析完成!")
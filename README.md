import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================== 参数设置 ==================
input_file = '/dat05/users/dinghao/h_or/jieguo/vae_latent_3000_cpu.xlsx'
sheet_name = 'Sheet1'
id_column_index = 0        # 第一列作为行名（Latent_Dim_X）
use_labels = False         # 是否有真实分组？没有就设为 False
n_neighbors = 10
min_dist = 0.09
random_state = 42
figsize = (12, 10)

# ================== 读取并预处理数据 ==================
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ 找不到文件: {input_file}")

# 读取 Excel
df = pd.read_excel(input_file, sheet_name=sheet_name)

print("✅ 原始数据前几行：")
print(df.head())

# 设置第一列为索引（如 Latent_Dim_1, Latent_Dim_2...）
df = df.set_index(df.columns[id_column_index])

# 转为数值型（跳过非数字）
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# 转置：现在每一行是一个样本（原列名 -1, 1, 3...），每一列是一个 latent 特征
data_transposed = df_numeric.T

print(f"\n✅ 转置后数据形状: {data_transposed.shape} (样本数 × 特征数)")
print("前几行预览（每个样本的 latent vector）：")
print(data_transposed.head())

# 检查是否为空
if data_transposed.empty or data_transposed.isna().all().all():
    raise ValueError("❌ 转置后无有效数值数据，请检查表格内容是否为纯数字。")

# 提取特征用于 UMAP
feature_data = data_transposed.dropna(axis=1, how='all')  # 去掉全空列
print(f"📊 使用 {feature_data.shape[1]} 个 latent 维度进行 UMAP 降维")

# 标签：使用原始列名（-1, 1, 3...）作为样本标签
labels = data_transposed.index.astype(str)

# ================== UMAP 降维 ==================
print("🚀 正在进行 UMAP 降维...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric='euclidean',
    random_state=random_state
)
embedding = reducer.fit_transform(feature_data)

# 构建结果 DataFrame
umap_df = pd.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1],
    'Sample': labels
})

# ================== 绘图 ==================
print("📈 正在绘制 UMAP 结果...")
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=figsize)

# 使用分类颜色
palette = "tab10" if umap_df['Sample'].nunique() <= 10 else "Spectral"
ax = sns.scatterplot(
    x='UMAP1', y='UMAP2',
    hue='Sample',
    data=umap_df,
    palette=palette,
    s=120, alpha=0.85, edgecolor='k', linewidth=0.4
)

# 图表设置
plt.title('UMAP Projection of 213 Samples', fontsize=16, pad=20)
plt.xlabel('UMAP1', fontsize=12)
plt.ylabel('UMAP2', fontsize=12)

# 图例放在右边外侧
ax.legend(title='Sample ID', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

plt.tight_layout()
plt.show()

print("✅ UMAP 可视化完成！共降维了 213 个样本。")

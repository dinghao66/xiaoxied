#t-sne可视化
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_visualize_excel(file_path):
    # 读取Excel文件
    data = pd.read_excel(file_path)

   tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)

plt.figure(figsize=(10, 8))    
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])    
plt.title('t-SNE Visualization of Excel Data')    
plt.xlabel('t-SNE Component 1')    
plt.ylabel('t-SNE Component 2')    
plt.show()

if __name__ == "__main__":
    file_path = "your_excel_file.xlsx"  # 替换为你的Excel文件路径
    tsne_visualize_excel(file_path)


    
    features = data.iloc[:, 1:]

    # 假设数据的特征部分不包含第一列（如果第一列是标签列等情况）

    features = data.iloc[:, 1:]
    # 如果数据全是特征，可以使用下面这行
    # features = data

    # 初始化TSNE模型
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)

    # 对特征数据进行降维
    reduced_data = tsne.fit_transform(features)

    # 绘制t-SNE结果
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title('t-SNE Visualization of Excel Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()



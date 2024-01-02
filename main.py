import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

import warnings

warnings.filterwarnings('ignore')

# Đọc dữ liệu từ bộ dataset và hiển thị 5 hàng đầu của bộ dữ liệu
df = pd.read_csv('dataset/Mall_Customers.csv')
print(df.head())

# Các cột của bộ dữ liệu
print(df.columns)

# Thông tin chi tiết của từng cột
print(df.info())

# Thống kê mô tả
df_des = df.describe()
print(df_des)

# Kiểm tra ô trống
print(df.isnull().sum())

# Xóa cột CustomerID
df = df.drop(columns=['CustomerID'])
print(df)

# Vẽ biểu đồ thống kê mô tả cho cột Age
df_des['Age'].plot(kind='line', figsize=(8, 4), title='Age')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Vẽ biểu đồ thống kê mô tả cho cột Annual Income
df_des['Annual Income (k$)'].plot(kind='line', figsize=(8, 4), title='Annual Income (k$)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Vẽ biểu đồ thống kê mô tả cho cột Annual Income
df_des['Spending Score (1-100)'].plot(kind='line', figsize=(8, 4), title='Spending Score (1-100)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Vẽ biểu đồ Scatter Plot thể hiện sự tương quan dữ liệu giữa 2 cột Annual Income và Spending Score (1-100)
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Sự phân bổ dữ liệu Annual Income và Spending Score')
plt.show()

# Màu sắc cho các phần trong biểu đồ
colors = ['lightcoral', 'lightskyblue']

# Vẽ biểu đồ thể hiện tần suất của cột Gender (Giới tính)
plt.figure(figsize=(6, 6))
plt.pie(df['Gender'].value_counts(), labels=df['Gender'].value_counts().index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Biểu đồ thống kê giới tính (Gender)')
patches, texts = plt.pie(df['Gender'].value_counts(), colors=colors, startangle=90)
plt.legend(patches, df['Gender'].value_counts().index, loc="best")
plt.tight_layout()
plt.show()

# Vẽ biểu đồ phân phối cho các cột dữ liệu
plt.figure(1, figsize=(15, 6))
i = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    i += 1
    plt.subplot(1, 3, i)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.distplot(df[x], bins=15)
    plt.title('Biểu đồ phân phối của {}'.format(x))
plt.show()

# Vẽ biểu đồ thể hiện tần xuất các cột dữ liệu
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

i = 0
for column in df.columns:
    row = i // 2
    col = i % 2
    ax = axs[row, col]

    counts = df[column].value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_title(f'Tần suất của {column}')
    ax.set_xlabel('Giá trị')
    ax.set_ylabel('Số lần xuất hiện')
    ax.grid(True)
    i += 1

plt.tight_layout()
plt.show()

# Hiển thị sự phân bố của các điểm dữ liệu theo giới tính
sns.pairplot(df, vars=['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue="Gender")
plt.show()

# Vẽ biểu đồ tổng quan về kiểm tra các giá trị ngoại lệ của các cột dữ liệu
plt.figure(figsize=(10, 6))
i = 0
for column in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    plt.subplot(1, 3, i + 1)
    sns.boxplot(df[column], color='white')
    plt.grid(True)
    plt.title(f'{column}')
    i += 1

plt.tight_layout()
plt.show()

# Chuẩn hóa dữ liệu để dữ liệu trở nên đồng đều hơn, cái thiện hiệu suất mô hình
# Khởi tạo StandardScaler
scaler = StandardScaler()

# Tên các cột chuẩn hóa
columns_to_scale = ['Annual Income (k$)', 'Spending Score (1-100)']

# Chuẩn hóa dữ liệu của các cột đã chọn
X = scaler.fit_transform(df[columns_to_scale])
print(X)

# Duyệt từ 1 -> 14 để tìm k thích hợp
inertia = []
for n in range(1, 15):
    algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                       tol=0.0001, random_state=111, algorithm='elkan')
    algorithm.fit(X)
    inertia.append(algorithm.inertia_)

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 15), inertia, 'o', color='blue')
plt.plot(np.arange(1, 15), inertia, '-', alpha=0.5, color='orange')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Tiến hành xác nhận xem k tìm được có ổn không
model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(1, 15))

visualizer.fit(X)
visualizer.show()
plt.show()

# Áp dụng KMeans với k=5
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, random_state=111, algorithm='elkan')
kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
df['Cluster'] = pd.DataFrame(y_kmeans)
print(df)

# Hiển thị dữ liệu sau khi phân cụm bằng màu sắc tương ứng và vẽ các tâm cụm
plt.figure(figsize=(8, 6))

unique_labels = np.unique(y_kmeans)
for label in unique_labels:
    plt.scatter(X[y_kmeans == label, 0], X[y_kmeans == label, 1], label=f'Cụm {label + 1}', edgecolors='black')
plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='red', s=100, edgecolors='black')
plt.title('Biểu đồ phân cụm dữ liệu')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# Tính độ đo Silhouette Score của mô hình với k từ 2 -> 14
for n in range(2, 15):
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    y_kmeans = kmeans.fit_predict(X)
    print(f'Silhouette Score (k = {n}): {silhouette_score(X, y_kmeans)}')

# Xuất dữ liệu đã được phân cụm ra 1 file mới
df.to_csv("export/Mall_Customers_ok.csv", index=False)

import pandas as pd

from data_processing.data_loader import load_data, drop_column
from data_visualization.visualizer import *
from data_analysis.preprocessing import standardize_data
from data_analysis.clustering import *

import warnings

warnings.filterwarnings('ignore')


def main():
    # Đọc bộ dữ liệu ra và hiển thị lên màn hình
    file_path = 'dataset/Mall_Customers.csv'
    df = load_data(file_path)
    print('[DỮ LIỆU GỐC]')
    print(df.head())

    # Các cột của bộ dữ liệu
    print('\n[LIỆT KÊ CÁC CỘT CỦA BỘ DỮ LIỆU]')
    print(df.columns)

    # Thông tin chi tiết của từng cột
    print('\n[THÔNG TIN CÁC CỘT CỦA BỘ DỮ LIỆU]')
    print(df.info())

    # Thống kê mô tả
    df_des = df.describe()
    print('\n[THỐNG KÊ MÔ TẢ CỦA BỘ DỮ LIỆU]')
    print(df_des)

    # Kiểm tra ô trống
    print('\n[KIỂM TRA GIÁ TRỊ TRỐNG CỦA TỪNG CỘT]')
    print(df.isnull().sum())

    # Xóa cột dữ liệu nhiễu
    columns_to_drop = ['CustomerID']
    df = drop_column(df, columns_to_drop)
    print('\n[DỮ LIỆU SAU KHI XÓA CỘT DỮ LIỆU NHIỄU]')
    print(df)

    # Trực quan hóa các cột dữ liệu bằng đồ thị
    line_plot(df_des['Age'], 'Age', 'Thống kê mô tả cột dữ liệu Age')
    line_plot(df_des['Annual Income (k$)'], 'Annual Income (k$)', 'Thống kê mô tả cột dữ liệu Annual Income (k$)')
    line_plot(df_des['Spending Score (1-100)'], 'Spending Score (1-100)',
              'Thống kê mô tả cột dữ liệu Spending Score (1-100)')
    scatter_plot(df['Annual Income (k$)'], df['Spending Score (1-100)'], 'Annual Income', 'Spending Score',
                 'Sự phân bổ dữ liệu Annual Income và Spending Score')
    pie_chart(df['Gender'], 'Biểu đồ thống kê giới tính (Gender)')

    for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        dist_plot(df, x)

    bar_plot(df, columns=df.columns)
    pair_plot(df, columns=['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue_col="Gender")
    box_plot(df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    # Chuẩn hóa dữ liệu
    columns_to_scale = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = standardize_data(df, columns_to_scale)
    print('\n[DỮ LIỆU CHUẨN HÓA CỘT Annual Income (k$) và Spending Score (1-100)]')
    print(X)

    # Tiến hành tìm k
    optimal_k = find_optimal_k(X)
    print('\n[QUÁN TÍNH CỦA MỖI GIÁ TRỊ K]')
    print(optimal_k)

    # Vẽ đồ thị thể hiện độ quán tính của mỗi giá trị k
    plot_inertia(1, 15, optimal_k, ['o', '-'], ['blue', 'orange'], 'Number of Clusters', 'Inertia',
                 'Đồ thị thể hiện độ quán tính tương ứng với số cụm')

    # Đánh giá k đã chọn
    visualizer = check_k_finded(X)
    visualizer.show()
    plt.show()

    # Thông qua việc đánh giá ta đã tìm được k hợp lý, tiến hành phân cụm
    k = 5
    y_kmeans, centers = apply_kmeans(X, k)
    print("\nNhãn của các cụm:", y_kmeans)
    print("\nTọa độ tâm của các cụm:", centers)

    df['Cluster'] = pd.DataFrame(y_kmeans)
    print('\n[BỘ DỮ LIỆU SAU KHI ĐƯỢC PHÂN CỤM]')
    print(df)

    # Vẽ đồ thị bộ dữ liệu sau khi phân cụm
    plot_cluster_data(X, y_kmeans, centers, 'Annual Income', 'Spending Score', 'Biểu đồ phân cụm dữ liệu', 'black',
                      'black')

    # Đánh giá kết quả phân cụm
    silhouette_scores(X)

    # Xuất bộ dữ liệu đã phân cụm
    output_file = "export/Mall_Customers_ok.csv"
    export_clustered_data(df, output_file)


if __name__ == "__main__":
    main()

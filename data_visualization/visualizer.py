import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def line_plot(data, x, title):
    data.plot(kind='line', x=x, figsize=(8, 4), title=title)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.show()


def scatter_plot(x, y, xlabel, ylabel, title):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def pie_chart(data_col, title):
    plt.figure(figsize=(6, 6))
    plt.pie(data_col.value_counts(), labels=data_col.value_counts().index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(title)
    patches, texts = plt.pie(data_col.value_counts(), colors=['lightcoral', 'lightskyblue'], startangle=90)
    plt.legend(patches, data_col.value_counts().index, loc="best")
    plt.tight_layout()
    plt.show()


def dist_plot(data, column):
    plt.figure(figsize=(8, 4))
    sns.distplot(data[column], bins=15)
    plt.title('Biểu đồ phân phối của {}'.format(column))
    plt.show()


def bar_plot(data, columns):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    i = 0
    for column in columns:
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        counts = data[column].value_counts().sort_index()
        ax.bar(counts.index, counts.values)
        ax.set_title(f'Tần suất của {column}')
        ax.set_xlabel('Giá trị')
        ax.set_ylabel('Số lần xuất hiện')
        ax.grid(True)
        i += 1

    plt.tight_layout()
    plt.show()


def pair_plot(data, columns, hue_col):
    sns.pairplot(data, vars=columns, hue=hue_col)
    plt.show()


def box_plot(data, columns):
    plt.figure(figsize=(10, 6))
    i = 0
    for column in columns:
        plt.subplot(1, 3, i + 1)
        sns.boxplot(data[column], color='white')
        plt.grid(True)
        plt.title(f'{column}')
        i += 1

    plt.tight_layout()
    plt.show()


def plot_inertia(a_start, a_end, inertia, marker, colors, xlabel, ylabel, title):
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(a_start, a_end), inertia, marker[0], color=colors[0])
    plt.plot(np.arange(a_start, a_end), inertia, marker[1], alpha=0.5, color=colors[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_cluster_data(X, y_kmeans, centers, xlabel, ylabel, title, data_edgecolors, cluster_edgecolors):
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(y_kmeans)
    for label in unique_labels:
        plt.scatter(X[y_kmeans == label, 0], X[y_kmeans == label, 1], label=f'Cụm {label + 1}',
                    edgecolors=cluster_edgecolors)
    plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='red', s=100, edgecolors=data_edgecolors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

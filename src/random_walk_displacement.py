import numpy as np
import matplotlib.pyplot as plt


def random_walk_displacement(num_steps, num_simulations):
    """
    模拟随机行走并返回每次模拟的最终位移

    参数:
    num_steps (int): 随机行走的步数
    num_simulations (int): 模拟的次数

    返回:
    np.ndarray: 包含每次模拟最终位移的二维数组，每一行代表一个维度上的位移
    """
    # 生成随机行走的位移矩阵，2 代表两个维度（如 x 和 y 方向）
    displacements_matrix = np.random.choice([-1, 1], size=(2, num_simulations, num_steps))
    # 对步数维度进行求和，得到每个模拟在两个维度上的最终位移
    final_displacements = np.sum(displacements_matrix, axis=2)
    return final_displacements


def plot_displacement_distribution(final_displacements, bins=30):
    """
    绘制位移分布直方图

    参数:
    final_displacements (np.ndarray): 包含每次模拟最终位移的二维数组
    bins (int): 直方图的组数
    """
    #即最终位移的大小
    magnitudes = np.sqrt(final_displacements[0] ** 2 + final_displacements[1] ** 2)
    #创建直方图
    plt.hist(magnitudes, bins=bins, density=True, alpha=0.7, color='b')
    #直方图标题
    plt.title('Random Walk Displacement Distribution')
    #为X轴贴标签
    plt.xlabel('Final Displacement')
    #为Y轴贴标签
    plt.ylabel('Probability Density')
    #设置网格线
    plt.grid(True)
    plt.show()


def plot_displacement_square_distribution(final_displacements, bins=30):
    """
    绘制位移平方分布直方图

    参数:
    final_displacements (np.ndarray): 包含每次模拟最终位移的二维数组
    bins (int): 直方图的组数
    """
    # 计算每个模拟的最终位移的平方
    squares = final_displacements[0] ** 2 + final_displacements[1] ** 2
    plt.hist(squares, bins=bins, density=True, alpha=0.7, color='b')
    plt.title('Random Walk Displacement Square Distribution')
    plt.xlabel('Final Displacement Square')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 可调整的参数
    num_steps = 1000  # 随机行走的步数
    num_simulations = 1000  # 模拟的次数
    bins = 30  # 直方图的组数

    # 模拟随机行走
    displacements = random_walk_displacement(num_steps, num_simulations)

    # 绘制位移分布直方图
    plot_displacement_distribution(displacements, bins)

    # 绘制位移平方分布直方图
    plot_displacement_square_distribution(displacements, bins)

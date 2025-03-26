import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def random_walk_finals(num_steps, num_walks):
    """生成多个二维随机游走的终点位置"""
    # 初始化x和y坐标
    x = np.zeros((num_walks, num_steps))
    y = np.zeros((num_walks, num_steps))
    
    # 生成随机步长（-1或1）
    x_steps = np.random.choice([-1, 1], size=(num_walks, num_steps))
    y_steps = np.random.choice([-1, 1], size=(num_walks, num_steps))
    
    # 计算累积位移
    x = np.cumsum(x_steps, axis=1)
    y = np.cumsum(y_steps, axis=1)
    
    # 提取最终终点坐标
    x_finals = x[:, -1]
    y_finals = y[:, -1]
    
    return x_finals, y_finals

def plot_endpoints_distribution(endpoints):
    """绘制二维随机游走终点的散点图"""
    x_coords, y_coords = endpoints
    plt.scatter(x_coords, y_coords, alpha=0.5, s=10, color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Random walk endpoint distribution")
    plt.axis("equal")

def analyze_x_distribution(endpoints):
    """分析随机游走终点x坐标的统计特性"""
    x_coords, _ = endpoints
    mean_x = np.mean(x_coords)
    var_x = np.var(x_coords, ddof=1)  # 使用n-1计算样本方差
    
    # 绘制直方图
    plt.hist(x_coords, bins=30, density=True, alpha=0.6, color='g')
    
    # 绘制正态分布拟合曲线
    x_vals = np.linspace(min(x_coords), max(x_coords), 100)
    pdf = norm.pdf(x_vals, loc=mean_x, scale=np.sqrt(var_x))
    plt.plot(x_vals, pdf, 'r', linewidth=2)
    
    plt.xlabel("X")
    plt.ylabel("probability density")
    plt.title(f"XCoordinate distribution (mean value={mean_x:.2f}, variance={var_x:.2f})")
    
    return mean_x, var_x

if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以保证可重复性
    
    # 生成数据
    endpoints = random_walk_finals(1000, 1000)
    
    # 创建图形
    plt.figure(figsize=(12, 5))
    
    # 绘制终点分布
    plt.subplot(121)
    plot_endpoints_distribution(endpoints)
    
    # 分析x坐标分布
    plt.subplot(122)
    analyze_x_distribution(endpoints)
    
    plt.tight_layout()
    plt.show()
pass
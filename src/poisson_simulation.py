import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def plot_poisson_pmf(lambda_param=8, max_l=20):
    """绘制泊松分布的概率质量函数
    
    参数:
        lambda_param (float): 泊松分布参数λ
        max_l (int): 最大的l值
    """
    l_values = np.arange(0, max_l + 1)
    pmf_values = (np.exp(-lambda_param) * lambda_param**l_values) / factorial(l_values)
    
    plt.bar(l_values, pmf_values, color='blue', alpha=0.7, label=f'λ={lambda_param}')
    plt.xlabel('l')
    plt.ylabel('P(l)')
    plt.title('Poisson PMF')
    plt.legend()
    plt.grid(True)

def simulate_coin_flips(n_experiments=10000, n_flips=100, p_head=0.08):
    """模拟多组抛硬币实验
    
    参数:
        n_experiments (int): 实验组数N
        n_flips (int): 每组抛硬币次数
        p_head (float): 正面朝上的概率
        
    返回:
        ndarray: 每组实验中正面朝上的次数
    """
    results = np.random.binomial(n=n_flips, p=p_head, size=n_experiments)
    return results

def compare_simulation_theory(n_experiments=10000, lambda_param=8):
    """比较实验结果与理论分布
    
    参数:
        n_experiments (int): 实验组数
        lambda_param (float): 泊松分布参数λ
    """
    # 实验结果
    simulated_results = simulate_coin_flips(n_experiments=n_experiments, n_flips=100, p_head=lambda_param / 100)
    unique, counts = np.unique(simulated_results, return_counts=True)
    simulated_pmf = counts / n_experiments

    # 理论分布
    max_l = max(unique)
    l_values = np.arange(0, max_l + 1)
    theoretical_pmf = (np.exp(-lambda_param) * lambda_param**l_values) / factorial(l_values)

    # 绘制对比图
    plt.bar(unique, simulated_pmf, color='orange', alpha=0.7, label='Simulation')
    plt.plot(l_values, theoretical_pmf, 'b-', marker='o', label='Theory')
    plt.xlabel('l')
    plt.ylabel('Probability')
    plt.title('Simulation vs Theory')
    plt.legend()
    plt.grid(True)

    # 打印统计信息
    mean_simulated = np.mean(simulated_results)
    var_simulated = np.var(simulated_results)
    print(f"Simulated Mean: {mean_simulated}, Simulated Variance: {var_simulated}")
    print(f"Theoretical Mean: {lambda_param}, Theoretical Variance: {lambda_param}")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 绘制理论分布
    plot_poisson_pmf()
    
    # 2&3. 进行实验模拟并比较结果
    compare_simulation_theory()
    
    plt.show()
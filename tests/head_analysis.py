import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 解析数据函数
def parse_head_data(data_text):
    """解析头部开关数据"""
    layers = []
    current_layer = None
    
    lines = data_text.strip().split('\n')
    for line in lines:
        if 'head allocate:' in line:
            # 提取头部分配数据
            start_idx = line.find('head allocate:') + len('head allocate: ')
            data_str = line[start_idx:].strip()
            
            # 解析为列表
            data = eval(data_str)
            # 提取32个头的数据
            heads = [item[0] for item in data[0]]
            layers.append(heads)
    
    return np.array(layers, dtype=float)

task1_name = "Single QA Training" # Single QA, Summarization, MultiHop QA
task2_name = "Single QA Training2" 

# Single QA数据（从你的输出中提取）
single_qa_text = """
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
"""

# Summarization数据（从你的输出中提取）
summarization_text = """
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
task_ids: None, head allocate: [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
"""

# 解析数据
single_qa_data = parse_head_data(single_qa_text)
summarization_data = parse_head_data(summarization_text)

print(f"Single QA数据形状: {single_qa_data.shape}")
print(f"Summarization数据形状: {summarization_data.shape}")

# 计算相似度分析
def analyze_similarity(data1, data2, task1_name, task2_name):
    """分析两个任务之间的相似度"""
    print(f"\n=== {task1_name} vs {task2_name} 相似度分析 ===")
    
    # 1. 整体相似度（Jaccard相似度）
    overall_sim = np.mean(data1 == data2)
    print(f"整体相似度（相同位置的比例）: {overall_sim:.4f}")
    
    # 2. 层级相似度
    layer_sims = []
    for layer_idx in range(min(len(data1), len(data2))):
        layer_sim = np.mean(data1[layer_idx] == data2[layer_idx])
        layer_sims.append(layer_sim)
        print(f"  层 {layer_idx+1}: {layer_sim:.4f}")
    
    # 3. 激活头数量统计
    active_heads_task1 = np.sum(data1)
    active_heads_task2 = np.sum(data2)
    total_heads = data1.size
    
    print(f"\n激活头数量统计:")
    print(f"  {task1_name}: {active_heads_task1}/{total_heads} ({active_heads_task1/total_heads:.2%})")
    print(f"  {task2_name}: {active_heads_task2}/{total_heads} ({active_heads_task2/total_heads:.2%})")
    
    # 4. 共同激活的头
    common_active = np.sum((data1 == 1) & (data2 == 1))
    only_task1 = np.sum((data1 == 1) & (data2 == 0))
    only_task2 = np.sum((data1 == 0) & (data2 == 1))
    
    print(f"\n激活头重叠分析:")
    print(f"  共同激活的头: {common_active} ({common_active/total_heads:.2%})")
    print(f"  仅{task1_name}激活: {only_task1} ({only_task1/total_heads:.2%})")
    print(f"  仅{task2_name}激活: {only_task2} ({only_task2/total_heads:.2%})")
    
    return overall_sim, layer_sims

# 执行相似度分析
overall_sim, layer_sims = analyze_similarity(single_qa_data, summarization_data, 
                                             task1_name, task2_name)

# 可视化1: 两个任务的热图对比
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Single QA热图
im1 = axes[0, 0].imshow(single_qa_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
axes[0, 0].set_title(task1_name + ' - Head Activation Pattern', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Head Index', fontsize=12)
axes[0, 0].set_ylabel('Layer Index', fontsize=12)
axes[0, 0].set_xticks(range(0, 32, 4))
axes[0, 0].set_yticks(range(len(single_qa_data)))
plt.colorbar(im1, ax=axes[0, 0], label='Activation (0=off, 1=on)')

# 2. Summarization热图
im2 = axes[0, 1].imshow(summarization_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
axes[0, 1].set_title(task2_name + ' - Head Activation Pattern', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Head Index', fontsize=12)
axes[0, 1].set_ylabel('Layer Index', fontsize=12)
axes[0, 1].set_xticks(range(0, 32, 4))
axes[0, 1].set_yticks(range(len(summarization_data)))
plt.colorbar(im2, ax=axes[0, 1], label='Activation (0=off, 1=on)')

# 3. 差异热图
diff_data = np.abs(single_qa_data - summarization_data)
im3 = axes[1, 0].imshow(diff_data, aspect='auto', cmap='Reds', vmin=0, vmax=1)
axes[1, 0].set_title('Difference Map (1 = Different, 0 = Same)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Head Index', fontsize=12)
axes[1, 0].set_ylabel('Layer Index', fontsize=12)
axes[1, 0].set_xticks(range(0, 32, 4))
axes[1, 0].set_yticks(range(len(diff_data)))
plt.colorbar(im3, ax=axes[1, 0], label='Difference')

# 4. 层相似度柱状图
layer_indices = np.arange(1, len(layer_sims) + 1)
bars = axes[1, 1].bar(layer_indices, layer_sims)
axes[1, 1].set_title('Layer-wise Similarity', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Layer Index', fontsize=12)
axes[1, 1].set_ylabel('Similarity', fontsize=12)
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].axhline(y=overall_sim, color='r', linestyle='--', label=f'Overall: {overall_sim:.3f}')
axes[1, 1].legend()

# 为相似度低的层添加特殊标记
for i, sim in enumerate(layer_sims):
    if sim < 0.5:
        bars[i].set_color('red')
    elif sim > 0.9:
        bars[i].set_color('green')

plt.tight_layout()
plt.savefig('head_activation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 可视化2: 激活头统计
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 激活密度图
layer_density_qa = np.mean(single_qa_data, axis=1)
layer_density_sum = np.mean(summarization_data, axis=1)
layer_indices = np.arange(1, len(layer_density_qa) + 1)

axes[0].plot(layer_indices, layer_density_qa, 'o-', label=task1_name, linewidth=2, markersize=8)
axes[0].plot(layer_indices, layer_density_sum, 's-', label=task2_name, linewidth=2, markersize=8)
axes[0].set_title('Activation Density per Layer', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Layer Index', fontsize=12)
axes[0].set_ylabel('Activation Density', fontsize=12)
axes[0].set_xticks(layer_indices)
axes[0].set_ylim(0, 1.1)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 头激活类型分布
categories = ['Common Active', task1_name, task2_name, 'Both Inactive']
values = [
    np.sum((single_qa_data == 1) & (summarization_data == 1)),
    np.sum((single_qa_data == 1) & (summarization_data == 0)),
    np.sum((single_qa_data == 0) & (summarization_data == 1)),
    np.sum((single_qa_data == 0) & (summarization_data == 0))
]

colors = ['green', 'blue', 'orange', 'gray']
bars = axes[1].bar(categories, values, color=colors)
axes[1].set_title('Head Activation Type Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Heads', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# 在柱状图上添加数值
for bar, val in zip(bars, values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('head_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

# 可视化3: 详细差异分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 差异层统计
diff_counts = np.sum(diff_data, axis=1)
layer_indices = np.arange(1, len(diff_counts) + 1)

axes[0, 0].bar(layer_indices, diff_counts)
axes[0, 0].set_title('Number of Differing Heads per Layer', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Layer Index', fontsize=12)
axes[0, 0].set_ylabel('Number of Differing Heads', fontsize=12)
axes[0, 0].set_xticks(layer_indices)
axes[0, 0].set_ylim(0, 33)
axes[0, 0].axhline(y=16, color='r', linestyle='--', alpha=0.5, label='Half of heads (16)')

# 标记差异最大的层
max_diff_idx = np.argmax(diff_counts)
axes[0, 0].bar(max_diff_idx, diff_counts[max_diff_idx], color='red')
axes[0, 0].legend()

# 2. 激活模式的相关性
# 展平每层的激活模式并计算相关性
correlations = []
for i in range(min(len(single_qa_data), len(summarization_data))):
    corr = np.corrcoef(single_qa_data[i], summarization_data[i])[0, 1]
    correlations.append(corr if not np.isnan(corr) else 0)

axes[0, 1].plot(layer_indices, correlations, 'o-', linewidth=2, markersize=8)
axes[0, 1].set_title('Layer-wise Pattern Correlation', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Layer Index', fontsize=12)
axes[0, 1].set_ylabel('Correlation Coefficient', fontsize=12)
axes[0, 1].set_xticks(layer_indices)
axes[0, 1].set_ylim(-1.1, 1.1)
axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0, 1].grid(True, alpha=0.3)

# 3. 任务特定头识别
# 找出主要差异的头
task_specific_heads = []
for head_idx in range(32):
    qa_activation = np.mean(single_qa_data[:, head_idx])
    sum_activation = np.mean(summarization_data[:, head_idx])
    
    if abs(qa_activation - sum_activation) > 0.3:  # 差异显著
        task_specific_heads.append((head_idx, qa_activation, sum_activation))

task_specific_heads.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)

head_indices = [h[0] for h in task_specific_heads[:10]]
qa_acts = [h[1] for h in task_specific_heads[:10]]
sum_acts = [h[2] for h in task_specific_heads[:10]]

x = np.arange(len(head_indices))
width = 0.35

axes[1, 0].bar(x - width/2, qa_acts, width, label=task1_name, alpha=0.8)
axes[1, 0].bar(x + width/2, sum_acts, width, label=task2_name, alpha=0.8)
axes[1, 0].set_title('Top 10 Task-Specific Heads', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Head Index', fontsize=12)
axes[1, 0].set_ylabel('Activation Rate', fontsize=12)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f'H{h}' for h in head_indices])
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. 头激活稳定性分析
# 计算每个头在不同层中的激活稳定性
head_stability_qa = np.std(single_qa_data, axis=0)
head_stability_sum = np.std(summarization_data, axis=0)

axes[1, 1].plot(range(32), head_stability_qa, 'o-', label=task1_name, alpha=0.7)
axes[1, 1].plot(range(32), head_stability_sum, 's-', label=task2_name, alpha=0.7)
axes[1, 1].set_title('Head Activation Stability Across Layers', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Head Index', fontsize=12)
axes[1, 1].set_ylabel('Standard Deviation', fontsize=12)
axes[1, 1].set_xticks(range(0, 32, 4))
axes[1, 1].set_ylim(0, 0.6)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 分析报告
print("\n" + "="*80)
print("分析报告摘要")
print("="*80)

print(f"1. 整体相似度: {overall_sim:.4f}")
print(f"   两个任务的头部激活模式非常相似 ({overall_sim*100:.1f}% 相同)")

print(f"\n3. 激活头统计:")
print(f"   Single QA激活率: {np.mean(single_qa_data):.2%}")
print(f"   Summarization激活率: {np.mean(summarization_data):.2%}")

# 输出关键差异层
print("\n关键差异层（相似度<80%）:")
for i, sim in enumerate(layer_sims):
    if sim < 0.8:
        print(f"  层 {i+1}: 相似度 = {sim:.4f}, 差异头数 = {np.sum(diff_data[i])}")
        
        # 找出该层具体差异
        layer_diff = diff_data[i]
        diff_indices = np.where(layer_diff == 1)[0]
        if len(diff_indices) > 0:
            print(f"    差异头索引: {diff_indices.tolist()}")
            print(f"    QA激活: {single_qa_data[i, diff_indices].tolist()}")
            print(f"    Summarization激活: {summarization_data[i, diff_indices].tolist()}")
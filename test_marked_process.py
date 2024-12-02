import numpy as np
from generation import MarkedIntensityIndepenent, IntensitySumGaussianKernel, generate_samples_marked

# 创建一个二维的标记强度函数
marked_intensity = MarkedIntensityIndepenent(dim=2)

# 为每个维度创建一个高斯核和的强度函数
# 维度0：在t=5和t=15处有峰值
intensity1 = IntensitySumGaussianKernel(k=2, centers=[5, 15], stds=[2, 2], coefs=[1, 1])

# 维度1：在t=8和t=12处有峰值，强度更大
intensity2 = IntensitySumGaussianKernel(k=2, centers=[8, 12], stds=[2, 2], coefs=[2, 2])

# 初始化每个维度的强度函数
marked_intensity.initialize(intensity1, dim=0)
marked_intensity.initialize(intensity2, dim=1)

# 生成样本
T = 20.0  # 时间上限
n = 5     # 生成5个序列

sequences = generate_samples_marked(marked_intensity, T, n)

# 打印结果
for i, seq in enumerate(sequences):
    print(f"\n序列 {i+1}:")
    for event in seq:
        print(f"维度: {event[0]}, 时间: {event[1]:.2f}")

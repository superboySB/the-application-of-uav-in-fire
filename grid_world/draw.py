from env import *
from llm import *
from user_prompt_generation import *
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os



uav0_avg = 0.5909254128143333

uav1_avg = 0.759165167301501

uav2_avg = 0.7238400282172347


print('uav0:{}, uav1:{}, uav2:{}, total: {}'.format(uav0_avg, uav1_avg, uav2_avg,
                                                        uav0_avg + uav1_avg + uav2_avg))

uav0_var = 0.011737916745441088
uav1_var = 0.00962201717442039
uav2_var = 0.007224143282404739


    # 绘制条形图
labels = ['Random', 'CMAS', 'HMAS-2']
averages = [uav0_avg, uav1_avg, uav2_avg]
variances = [uav0_var, uav1_var, uav2_var]
print(uav0_var)
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, averages, yerr=np.sqrt(variances), capsize=10, color=['#FF6347', '#FF8C00', '#FFD700', '#FF4500'])

# 在条形图上显示具体数值
for bar, avg in zip(bars, averages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{avg:.2f}', ha='center', va='bottom')
#plt.xlabel('Framework')
plt.ylabel('Average Data Collection Ratio')
#plt.title('Average UAV Collection with Variance over 5 Episodes')
plt.savefig('uav_communication_framwork_comparison.png', bbox_inches='tight')
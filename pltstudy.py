import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
excel_file_path = 'lhm.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file_path)

# 提取横轴和纵轴数据
x_values = df['时间']  # 替换为你的横轴列名
y_values = df['是否攻击']  # 替换为你的纵轴列名
y1_values = df['实际标签']

# color_mapping = {
#     (0, 0.1): 'green',
#     (0.1, 1): 'yellow'
# }

# # 生成颜色列表
# colors = [next(color_mapping[k] for k in color_mapping if k[0] <= y <= k[1]) for y in y_values]
# 绘制柱状图
# plt.bar(x_values, [y - 0.1 for y in y_values],bottom=0.1, label='YourLabel')
bars = plt.bar(x_values, y_values, width = 50, label='PredictLabel')

# for bar, color in zip(bars, colors):
#     bar.set_color(color)
# 添加标签和标题
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# plt.ylim(bottom=0.1)
# 添加图例
plt.legend()

plt.plot(x_values, y1_values, linestyle='--', color='red', label='Your Data')
# 显示图形
custom_y_ticks = [0, 1]
custom_y_tick_labels = ['NoAttack', 'Attack']
plt.yticks(custom_y_ticks, custom_y_tick_labels)
plt.show()

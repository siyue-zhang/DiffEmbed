import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Data points
dream_x = [3200, 4800, 6400, 9600, 10880]
dream_y = [32.2, 39.1, 45.8, 44.8, 47.4]  # Replace with actual Dream data if available
qwen_x = [3200, 6400, 9600, 10880]
qwen_y = [36.4, 35.9, 38, 40.2]

# Formatter function for x-axis
def format_x_k(x, pos):
    return f'{int(x/1000)}k'

# Create the plot
plt.figure(figsize=(10, 7))

# Plot lines
plt.plot(dream_x, dream_y, label='Dream', color='purple', marker='o', markersize=16, linewidth=3)
plt.plot(qwen_x, qwen_y, label='Qwen2.5', color='blue', marker='o', markersize=16, linewidth=3)

# Apply formatter to x-axis
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_k))
plt.xticks([2000, 4000, 6000, 8000, 10000, 12000], fontsize=24)
plt.yticks([35, 40, 45, 50], fontsize=24)

plt.ylim([30,50])
# Axis labels
# plt.xlabel('Number of training samples', fontsize=20, labelpad=15)
plt.ylabel('TheoQ. nDCG@10(%)', fontsize=24, labelpad=20)

# Grid
plt.grid(color='lightgray', linestyle='--', linewidth=0.6)

# Legend
plt.legend(fontsize=24)

# Save to PDF
plt.tight_layout()
plt.savefig("new_plot.pdf")

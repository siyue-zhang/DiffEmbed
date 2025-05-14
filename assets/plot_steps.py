import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Data points
dream_x = [3200, 6400, 9600, 10880]
dream_y = [34, 37, 42, 48.3]  # Replace with actual Dream data if available
qwen_x = [3200, 6400, 9600, 10880]
qwen_y = [36.4, 35.9, 38, 40.2]

# Formatter function for x-axis
def format_x_k(x, pos):
    return f'{int(x/1000)}k'

# Create the plot
plt.figure(figsize=(8, 6))

# Plot lines
plt.plot(dream_x, dream_y, label='Dream', color='purple', marker='o', markersize=8)
plt.plot(qwen_x, qwen_y, label='Qwen2.5', color='blue', marker='o', markersize=8)

# Apply formatter to x-axis
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_k))
plt.xticks([2000, 4000, 6000, 8000, 10000], fontsize=16)
plt.yticks([30, 35, 40, 45, 50], fontsize=16)

# Axis labels
plt.xlabel('Number of training samples', fontsize=18, labelpad=15)
plt.ylabel('TheoQ. nDCG@10(%)', fontsize=18, labelpad=15)

# Grid
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

# Legend
plt.legend(fontsize=18)

# Save to PDF
plt.tight_layout()
plt.savefig("plot.pdf")

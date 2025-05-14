import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Define the same instances
instances = ["Vieta's Formulas", "Pigeonhole Principle", "Euler's Identity", "Central Limit Theorem",
             "Two Pointers", "N-Queens Problem", "Sweep Line Algorithm", "Kahn's Algorithm"]

# Create a simple plot with 2 rows and 4 columns
plt.figure(figsize=(7.5, 3)) # Adjusted height for 2 rows
colors = cm.rainbow(np.linspace(0, 1, len(instances)))

# Create grid positions
x = np.array([0, 1, 2, 3] * 2)  # Column positions
y = np.array([0] * 4 + [1] * 4)  # Row positions

# Plot points and labels
for i, (instance, color) in enumerate(zip(instances, colors)):
    plt.scatter(x[i], y[i], color=color, s=300)  # Increased point size for better visibility
    
    # Split long instance names
    if len(instance) > 10:
        words = instance.split()
        mid = len(words) // 2
        label_text = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
    else:
        label_text = instance
    
    # Adjust xytext offset and text alignment for better separation
    plt.annotate(label_text, 
                (x[i], y[i]), 
                xytext=(0, -15 if y[i] == 0 else 15),  # Increased vertical separation
                textcoords='offset points',
                ha='center',
                va='top' if y[i] == 0 else 'bottom',  # Flipped vertical alignment
                rotation=30,
                fontsize=16)

plt.ylim(-0.5, 1.5)
plt.xlim(-0.5, 3.5)
plt.axis('off')
plt.tight_layout()
plt.savefig("color_mapping.pdf", dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()
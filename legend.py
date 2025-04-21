import matplotlib.pyplot as plt

# Define disease labels and corresponding colors, including "No Disease" as grey
disease_mapping = {
    0: "Anx",
    1: "OCD",
    2: "ADHD",
    3: "ODD",
    4: "Cond",
    5: "Specific Phobia",
    6: "HC"
}

colors = ['#1C9DFC', '#572BB6','#FE0303', '#F5510A', '#E8F60C','#4E4E49','#CCDEDE']
# Create a dummy plot for legend
fig, ax = plt.subplots(figsize=(2, 5))  # Adjust figure size to reduce whitespace
for color, (idx, label) in zip(colors, disease_mapping.items()):
    ax.plot([], [], marker='o', color=color, linestyle='None', markersize=10, label=label)

# Add and customize the legend
legend = ax.legend(loc='center', frameon=True, title="Disease Categories")

# Hide the plot axes to show only the legend
ax.axis('off')

# Adjust subplot parameters to minimize whitespace

plt.savefig('legend.png')
# Display the legend-only figure
plt.show()

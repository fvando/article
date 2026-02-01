import matplotlib.pyplot as plt
import pandas as pd
import os

# Data from Figure 9 but rounded
data = {
    "Description": [
        "Density check",
        "Model State",
        "Total workers needed",
        "Model Type",
        "Total Resolution Time",
        "Total Number of Iterations",
        "Number of Restrictions",
        "Number of Variables"
    ],
    "Value": [
        "final=0.35, threshold=0.36",
        "OPTIMAL",
        "95.10",
        "Linear or Integer Model",
        "1920 ms",
        "95",
        "7597",
        "9600"
    ]
}

df = pd.DataFrame(data)

# Create the figure
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Color the "OPTIMAL" cell green
for (row, col), cell in table.get_celld().items():
    if row == 2 and col == 1: # Row 1 (header is 0), Col 1 (Model State value)
        cell.set_facecolor('#d4edda') # Light green
        cell.get_text().set_color('#155724')
        cell.get_text().set_weight('bold')
    if row == 0:
        cell.set_facecolor('#f8f9fa')
        cell.get_text().set_weight('bold')

output_path = r"c:\project\article\thesis\PTVersion_v2\template\figuras\results.png"
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"Generated clean diagnostic table at {output_path}")

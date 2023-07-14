from nbformat import read, writes
from nbconvert.exporters.script import ScriptExporter

# Specify the path to your notebook file
notebook_file = 'final_project.ipynb'

# Read the notebook file
with open(notebook_file, 'r', encoding='utf-8') as f:
    nb = read(f, 4)

# Create a ScriptExporter instance
exporter = ScriptExporter()

# Export the notebook as a Python script
output, _ = exporter.from_notebook_node(nb)

# Save the Python script
with open('final_project.py', 'w', encoding='utf-8') as f:
    f.write(output)

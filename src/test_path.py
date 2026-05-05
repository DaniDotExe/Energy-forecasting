import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'EDA', 'SARIMAX_training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"Exists: {os.path.exists(OUTPUT_DIR)}")

plt.figure()
plt.plot([1, 2], [1, 2])
filepath = os.path.join(OUTPUT_DIR, 'test_plot.png')
print(f"Saving to: {filepath}")
plt.savefig(filepath)
print("Saved successfully.")

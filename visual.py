import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("isodata.csv")

y = df['Стоимость']
x1 = df['Симптомы_код']
x2 = df['Врач_код']
x3 = df['Анализы_код']

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(x1, y, color='red', alpha=0.7)
plt.xlabel('Симптомы_код')
plt.ylabel('Стоимость')
plt.title('X: Симптомы_код, Y: Стоимость')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(x2, y, color='green', alpha=0.7)
plt.xlabel('Врач_код')
plt.ylabel('Стоимость')
plt.title('X: Врач_код, Y: Стоимость')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(x3, y, color='red', alpha=0.7)
plt.xlabel('Анализы_код')
plt.ylabel('Стоимость')
plt.title('X: Анализы_код, Y: Стоимость')
plt.grid(True)

plt.tight_layout()
plt.show()
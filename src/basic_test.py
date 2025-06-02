import matplotlib.pyplot as plt
import numpy as np

print(f"Matplotlib backend: {plt.get_backend()}")

# Simple data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Simple Sine Wave Test")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
print("Calling plt.show()...")
plt.show()
print("plt.show() has finished.")
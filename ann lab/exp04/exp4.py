import numpy as np
import matplotlib.pyplot as plt

# Character A (5x5)

A = np.array([
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,0,0,1]
])

# Convert 0 → -1
pattern = np.where(A == 0, -1, 1)

# Flatten
pattern = pattern.flatten()

# Hopfield Weight Matrix
W = np.outer(pattern, pattern)

# No self-connection
np.fill_diagonal(W, 0)

# Create Distorted Character
distorted = pattern.copy()

# Flip some bits
distorted[0] *= -1
distorted[6] *= -1
distorted[18] *= -1
distorted[22] *= -1

# Recognition Process
recognized = distorted.copy()

for _ in range(10):
    recognized = np.sign(W @ recognized)

# Reshape
original_img = pattern.reshape(5,5)
distorted_img = distorted.reshape(5,5)
recognized_img = recognized.reshape(5,5)

# Display
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(original_img, cmap='binary')
plt.title("Stored A")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(distorted_img, cmap='binary')
plt.title("Distorted A")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(recognized_img, cmap='binary')
plt.title("Recognized A")
plt.axis('off')

plt.show()

print("Character Recognition Completed Successfully")
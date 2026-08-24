import numpy as np
import matplotlib.pyplot as plt

# Original Binary Image (5x5 Letter T)

original = np.array([
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0]
])

# Convert 0 → -1
pattern = np.where(original == 0, -1, 1)

# Flatten Pattern
pattern = pattern.flatten()

# Hopfield Weight Matrix
W = np.outer(pattern, pattern)

# Diagonal must be zero
np.fill_diagonal(W, 0)

# Create Noisy Image
noisy = pattern.copy()

# Flip some pixels
noisy[1] *= -1
noisy[7] *= -1
noisy[15] *= -1
noisy[22] *= -1

# Restore Image
restored = noisy.copy()

for _ in range(10):
    restored = np.sign(W @ restored)

# Convert back to 5x5
original_img = pattern.reshape(5,5)
noisy_img = noisy.reshape(5,5)
restored_img = restored.reshape(5,5)

# Display Images
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(original_img, cmap='binary')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(noisy_img, cmap='binary')
plt.title("Noisy")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(restored_img, cmap='binary')
plt.title("Restored")
plt.axis('off')

plt.show()
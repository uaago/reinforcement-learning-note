import numpy as np

np.random.seed(0)
probs = np.random.uniform(size=10)
print(probs)
print(np.argmax(probs))

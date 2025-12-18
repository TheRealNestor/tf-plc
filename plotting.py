# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
n_params = np.array([])
max_inference_time = np.array([])
memory_usage = np.array([])
# %%
plt.title("Model size effect on inference time")
plt.plot(n_params, max_inference_time)
plt.xlabel("No. parameters")
plt.ylabel("Max inference time [ms]")

# %%
plt.title("Model size effect on memory usage")
plt.plot(n_params, memory_usage)
plt.xlabel("No. parameters")
plt.ylabel("Memory usage [kB]")
# %%

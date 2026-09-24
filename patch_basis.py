import re
with open("bindings/python/qkrylov/basis.py", "r") as f:
    code = f.read()

# Change dtype defaults
code = code.replace("dtype: Any = np.float32", "dtype: Any = np.float64")
code = code.replace("dtype = np.float32", "dtype = np.float64")

# We need to implement list-to-vector for sz. Wait, the C++ Sector does it?
# Let's check `Sector` in python C++ binding. We just bound Sector. Does it take a vector?

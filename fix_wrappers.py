import os
import re

for file in ["bindings/python/qkrylov/site.py", "bindings/python/qkrylov/operators.py"]:
    with open(file, "r") as f:
        content = f.read()
    
    # Remove suffix = ... logic and replace getattr
    content = re.sub(r"suffix = '_FP64' if dtype == np\.float64 else '_FP32'\s+", "", content)
    content = re.sub(r"getattr\(_cpp, f'([A-Za-z0-9_]+)\{suffix\}'\)", r"getattr(_cpp, '\1')", content)
    
    with open(file, "w") as f:
        f.write(content)

import re

with open("bindings/python/qkrylov/site.py", "r") as f:
    code = f.read()

def replace_suffix(class_name):
    old = f'getattr(_cpp, f"{class_name}{{suffix}}")'
    new = f'(getattr(_cpp, "{class_name}") if hasattr(_cpp, "{class_name}") else getattr(_cpp, f"{class_name}{{suffix}}"))'
    return old, new

for name in ["SpinHalfSite", "SpinSSite", "FermionSite", "HubbardSite", "TJSite"]:
    old, new = replace_suffix(name)
    code = code.replace(old, new)

with open("bindings/python/qkrylov/site.py", "w") as f:
    f.write(code)

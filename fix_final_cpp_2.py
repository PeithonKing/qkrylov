with open("src/basis/fermion_basis.cpp", "r") as f: code = f.read()
code = code.replace("!sector_.use_n", "sector_.n.empty()")
with open("src/basis/fermion_basis.cpp", "w") as f: f.write(code)

with open("src/basis/hubbard_basis.cpp", "r") as f: code = f.read()
code = code.replace("!sector_.use_nup", "sector_.nup.empty()")
code = code.replace("!sector_.use_ndn", "sector_.ndn.empty()")
with open("src/basis/hubbard_basis.cpp", "w") as f: f.write(code)

with open("src/basis/tj_basis.cpp", "r") as f: code = f.read()
code = code.replace("!sector_.use_nup", "sector_.nup.empty()")
code = code.replace("!sector_.use_ndn", "sector_.ndn.empty()")
code = code.replace("!sector_.use_n", "sector_.n.empty()")
with open("src/basis/tj_basis.cpp", "w") as f: f.write(code)

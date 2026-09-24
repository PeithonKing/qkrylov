import re, glob

# Fix fermion_basis.cpp
with open("src/basis/fermion_basis.cpp", "r") as f: code = f.read()
code = code.replace("if(sector_.use_n)", "if(!sector_.n.empty())")
code = code.replace("if(popcount(s) == sector_.n)", "if(std::find(sector_.n.begin(), sector_.n.end(), popcount(s)) != sector_.n.end())")
with open("src/basis/fermion_basis.cpp", "w") as f: f.write(code)

# Fix hubbard_basis.cpp
with open("src/basis/hubbard_basis.cpp", "r") as f: code = f.read()
code = code.replace("if(sector_.use_nup)", "if(!sector_.nup.empty())")
code = code.replace("if(sector_.use_ndn)", "if(!sector_.ndn.empty())")
code = code.replace("if(popcount(s_up) == sector_.nup)", "if(std::find(sector_.nup.begin(), sector_.nup.end(), popcount(s_up)) != sector_.nup.end())")
code = code.replace("if(popcount(s_dn) == sector_.ndn)", "if(std::find(sector_.ndn.begin(), sector_.ndn.end(), popcount(s_dn)) != sector_.ndn.end())")
with open("src/basis/hubbard_basis.cpp", "w") as f: f.write(code)

# Fix tj_basis.cpp
with open("src/basis/tj_basis.cpp", "r") as f: code = f.read()
code = code.replace("if(sector_.use_nup)", "if(!sector_.nup.empty())")
code = code.replace("if(sector_.use_ndn)", "if(!sector_.ndn.empty())")
code = code.replace("if(popcount(s_up) == sector_.nup)", "if(std::find(sector_.nup.begin(), sector_.nup.end(), popcount(s_up)) != sector_.nup.end())")
code = code.replace("if(popcount(s_dn) == sector_.ndn)", "if(std::find(sector_.ndn.begin(), sector_.ndn.end(), popcount(s_dn)) != sector_.ndn.end())")
with open("src/basis/tj_basis.cpp", "w") as f: f.write(code)


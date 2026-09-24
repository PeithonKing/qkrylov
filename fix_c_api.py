import glob
import re

files = glob.glob("src/c_api*.cpp")

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    old_code = """    sec.use_sz = b.sector.use_sz;
    sec.sz2 = b.sector.sz2;
    sec.use_nup = b.sector.use_nup;
    sec.use_ndn = b.sector.use_ndn;
    sec.nup = b.sector.nup;
    sec.ndn = b.sector.ndn;
    sec.use_n = b.sector.use_n;
    sec.n = b.sector.n;
    sec.use_nb = b.sector.use_nb;
    sec.nb = b.sector.nb;"""

    new_code = """    if (b.sector.use_sz) sec.sz.push_back(b.sector.sz2 / 2.0);
    if (b.sector.use_nup) sec.nup.push_back(b.sector.nup);
    if (b.sector.use_ndn) sec.ndn.push_back(b.sector.ndn);
    if (b.sector.use_n) sec.n.push_back(b.sector.n);
    if (b.sector.use_nb) sec.nb.push_back(b.sector.nb);"""
    
    content = content.replace(old_code, new_code)
    
    with open(f, "w") as file:
        file.write(content)
print("Fixed C APIs")

with open("src/sites/spinhalf_site.cpp", "r") as f:
    lines = f.readlines()

# The file likely ends with
# }
# #include "qkrylov/operators/opsum.hpp"
# std::vector<Instruction> SpinHalfSite::compile ...

content = "".join(lines)
# I will just rewrite the whole file cleanly.
import re
# First, remove everything from #include "qkrylov/operators/opsum.hpp" onwards
idx = content.find('#include "qkrylov/operators/opsum.hpp"')
if idx != -1:
    content = content[:idx]

# Now, ensure it ends with } cleanly
content = content.strip()
if content.endswith('}'):
    content = content[:-1] # remove the closing brace of namespace

append_str = """
std::vector<Instruction> SpinHalfSite::compile(const OperatorTerm& term) const {
    std::vector<Instruction> insts;
    insts.push_back({0, 0, 0, 0, term.coeff});

    for (const auto& factor : term.factors) {
        std::vector<Instruction> next_insts;
        for (auto inst : insts) {
            uint64_t bit = 1ULL << factor.site;
            
            if (factor.op == "Sp") {
                if (inst.check_mask & bit) {
                    if (inst.expected_bits & bit) continue; // Requires UP, Sp is impossible
                }
                inst.check_mask |= bit;
                inst.expected_bits &= ~bit; // must be 0
                inst.flip_mask ^= bit;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Sm") {
                if (inst.check_mask & bit) {
                    if (!(inst.expected_bits & bit)) continue; // Requires DN, Sm is impossible
                }
                inst.check_mask |= bit;
                inst.expected_bits |= bit; // must be 1
                inst.flip_mask ^= bit;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Sz") {
                if (!(inst.check_mask & bit) || (inst.expected_bits & bit)) {
                    Instruction up = inst;
                    up.check_mask |= bit;
                    up.expected_bits |= bit;
                    up.coeff *= 0.5;
                    next_insts.push_back(up);
                }
                if (!(inst.check_mask & bit) || !(inst.expected_bits & bit)) {
                    Instruction dn = inst;
                    dn.check_mask |= bit;
                    dn.expected_bits &= ~bit;
                    dn.coeff *= -0.5;
                    next_insts.push_back(dn);
                }
            }
            else if (factor.op == "Sx") {
                inst.flip_mask ^= bit;
                inst.coeff *= 0.5;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Sy") {
                if (!(inst.check_mask & bit) || (inst.expected_bits & bit)) {
                    Instruction up = inst;
                    up.check_mask |= bit;
                    up.expected_bits |= bit;
                    up.flip_mask ^= bit;
                    up.coeff *= ComplexDouble(0.0, -0.5);
                    next_insts.push_back(up);
                }
                if (!(inst.check_mask & bit) || !(inst.expected_bits & bit)) {
                    Instruction dn = inst;
                    dn.check_mask |= bit;
                    dn.expected_bits &= ~bit;
                    dn.flip_mask ^= bit;
                    dn.coeff *= ComplexDouble(0.0, 0.5);
                    next_insts.push_back(dn);
                }
            }
            else {
                throw std::runtime_error("Unknown spin operator: " + factor.op);
            }
        }
        insts = std::move(next_insts);
    }
    return insts;
}

} // namespace qkrylov
"""

new_content = '#include "qkrylov/operators/opsum.hpp"\n' + content + append_str

with open("src/sites/spinhalf_site.cpp", "w") as f:
    f.write(new_content)

print("Fixed spinhalf_site.cpp")

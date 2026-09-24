import glob

for f in glob.glob("include/qkrylov/basis/*.hpp"):
    with open(f, "r") as file:
        lines = file.readlines()
    
    new_lines = []
    skip = 0
    for i, line in enumerate(lines):
        if skip > 0:
            skip -= 1
            continue
        # If it's the start of one of those constructors
        if "const sector::" in line:
            # We want to skip this line, the previous line if it was `    Name(` and the next lines until `) : Name(...) {}`
            # Actually, let's just use string replace on the whole text
            pass
        new_lines.append(line)
        
    with open(f, "w") as file:
        file.write("".join(new_lines))


import re
import glob

for f in glob.glob("include/qkrylov/basis/*.hpp"):
    with open(f, "r") as file:
        content = file.read()
    
    # Just delete any constructor taking `const sector::X&`
    content = re.sub(r'^\s*[A-Za-z0-9_]+\(\s*int\s+N,\s*(?:double\s+S,\s*)?const\s+sector::[A-Za-z0-9_]+&\s+[a-z]+\s*\)\s*:\s*[A-Za-z0-9_]+\([^\)]+\)\s*\{\}\n*', '', content, flags=re.MULTILINE|re.DOTALL)
    
    # Handle the ones with newline in arguments
    content = re.sub(r'^\s*[A-Za-z0-9_]+\(\s*int\s+N,\s*const\s+sector::[A-Za-z0-9_]+&\s+[a-z]+\s*\)\s*:\s*[A-Za-z0-9_]+\([^\)]+\)\s*\{\}\n*', '', content, flags=re.MULTILINE|re.DOTALL)

    # Some might just be declared but defined inline or not defined
    content = re.sub(r'\s*[A-Za-z0-9_]+\(\n\s*int\s+N,\n\s*const\s+sector::[A-Za-z0-9_]+&\s+[a-z]+\n\s*\)\s*:\s*[A-Za-z0-9_]+\([^\)]+\)\s*\{\}', '', content)
    
    with open(f, "w") as file:
        file.write(content)


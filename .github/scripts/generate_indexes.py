import os
import sys

def generate_index_html(package_name, wheels, repo_slug, tag):
    """
    Generates a PEP 503 compliant Simple Repository API HTML page.
    """
    html = f"<!DOCTYPE html>\n<html>\n<head>\n  <title>Links for {package_name}</title>\n</head>\n<body>\n  <h1>Links for {package_name}</h1>\n"
    for wheel in wheels:
        # Construct the GitHub Releases download URL
        url = f"https://github.com/{repo_slug}/releases/download/{tag}/{wheel}"
        html += f'  <a href="{url}">{wheel}</a><br/>\n'
    html += "</body>\n</html>\n"
    return html

def main():
    if len(sys.argv) != 4:
        print("Usage: generate_indexes.py <wheel_dir> <repo_slug> <tag>")
        sys.exit(1)

    wheel_dir = sys.argv[1]
    repo_slug = sys.argv[2]
    tag = sys.argv[3]
    package_name = "qkrylov"

    # Get all wheels in the directory
    wheels = [f for f in os.listdir(wheel_dir) if f.endswith('.whl')]

    if not wheels:
        print(f"No wheels found in {wheel_dir}")
        sys.exit(0)

    # We will simulate the PyTorch index structure:
    # We create index.html files in whl_out/cpu/qkrylov/index.html
    # and scaffolding for whl_out/cu12/qkrylov/index.html
    
    variants = ['cpu', 'cu12', 'rocm6']
    
    for variant in variants:
        # In the future, we would filter 'wheels' to only include those matching the variant (e.g. +cpu or +cu12).
        # For now, since we only build CPU wheels, we'll put them in all indices to test the scaffolding,
        # or just put them in 'cpu' and empty indices for the others.
        # Let's filter: if a wheel has a local version +cu12, it goes to cu12.
        # If it has no local version, it's cpu.
        
        variant_wheels = []
        for w in wheels:
            if f'+{variant}' in w:
                variant_wheels.append(w)
            elif variant == 'cpu' and '+' not in w:
                variant_wheels.append(w)
                
        out_dir = os.path.join("whl_out", variant, package_name)
        os.makedirs(out_dir, exist_ok=True)
        
        html_content = generate_index_html(package_name, variant_wheels, repo_slug, tag)
        with open(os.path.join(out_dir, "index.html"), "w") as f:
            f.write(html_content)
            
        print(f"Generated index for {variant} with {len(variant_wheels)} wheels.")

if __name__ == "__main__":
    main()

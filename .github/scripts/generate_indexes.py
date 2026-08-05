import os
import sys
from collections import defaultdict

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

def wheel_variant(wheel):
    parts = wheel.split("-")
    if len(parts) < 2:
        return None
    version = parts[1]
    if "+" not in version:
        return "cpu"
    return version.split("+", 1)[1]

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

    variants = defaultdict(list)
    for wheel in wheels:
        variant = wheel_variant(wheel)
        if variant:
            variants[variant].append(wheel)

    preferred_order = ["cpu", "cu12", "cu13", "rocm6"]
    ordered_variants = [
        variant for variant in preferred_order if variant in variants
    ] + sorted(
        variant for variant in variants if variant not in preferred_order
    )

    for variant in ordered_variants:
        variant_wheels = sorted(variants[variant])
                
        out_dir = os.path.join("whl_out", variant, package_name)
        os.makedirs(out_dir, exist_ok=True)
        
        html_content = generate_index_html(package_name, variant_wheels, repo_slug, tag)
        with open(os.path.join(out_dir, "index.html"), "w") as f:
            f.write(html_content)
            
        print(f"Generated index for {variant} with {len(variant_wheels)} wheels.")

if __name__ == "__main__":
    main()

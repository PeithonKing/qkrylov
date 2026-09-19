"""Auto-generate per-module Starlight Markdown pages for qkrylov Python bindings.

Uses Griffe AST static analysis with NumPy docstring parsing.
Produces one .md file per sub-module (solvers.md, basis.md, etc.)
plus an index.md that links to all of them.
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PY_BINDINGS = REPO_ROOT / "bindings" / "python"
OUT_DIR = REPO_ROOT / "docs" / "src" / "content" / "docs" / "api" / "auto" / "python"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import griffe
    from griffe import Parser
    USE_GRIFFE = True
except ImportError:
    USE_GRIFFE = False


def fmt_params(parameters) -> str:
    if not parameters:
        return "()"
    parts = []
    for p in parameters:
        part = str(p.name)
        if getattr(p, "annotation", None):
            part += f": {p.annotation}"
        if getattr(p, "default", None):
            part += f" = {p.default}"
        parts.append(part)
    return f"({', '.join(parts)})"


def safe_doc(obj) -> str:
    try:
        if getattr(obj, "docstring", None) is not None:
            return obj.docstring.value or ""
    except Exception:
        pass
    return ""


def render_function(name: str, fn, prefix: str = "qkrylov") -> list[str]:
    lines = []
    params = fmt_params(getattr(fn, "parameters", None))
    ret = f" -> {fn.returns}" if getattr(fn, "returns", None) else ""
    lines.append(f"### `{prefix}.{name}{params}{ret}`")
    lines.append("")
    doc = safe_doc(fn)
    if doc:
        lines.append(doc)
        lines.append("")
    return lines


def render_class(name: str, cls, prefix: str = "qkrylov") -> list[str]:
    lines = []
    lines.append(f"## `{prefix}.{name}`")
    lines.append("")
    doc = safe_doc(cls)
    if doc:
        lines.append(doc)
        lines.append("")
    methods = []
    if hasattr(cls, "members"):
        for m_name, m_val in cls.members.items():
            try:
                if not m_name.startswith("_") and getattr(m_val, "is_function", False):
                    methods.append((m_name, m_val))
            except Exception:
                continue
    if methods:
        lines.append("### Methods")
        lines.append("")
        for m_name, m_val in sorted(methods, key=lambda x: x[0]):
            params = fmt_params(getattr(m_val, "parameters", None))
            ret = f" -> {m_val.returns}" if getattr(m_val, "returns", None) else ""
            lines.append(f"#### `{m_name}{params}{ret}`")
            lines.append("")
            m_doc = safe_doc(m_val)
            if m_doc:
                lines.append(m_doc)
                lines.append("")
    return lines


def write_page(path: Path, title: str, description: str, body_lines: list[str]):
    content = ["---", f'title: "{title}"', f'description: "{description}"', "---", ""]
    content.extend(body_lines)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


if USE_GRIFFE:
    mod = griffe.load("qkrylov", search_paths=[str(PY_BINDINGS)], docstring_parser=Parser.numpy)

    submod_names = [
        k for k, v in mod.members.items()
        if not k.startswith("_") and getattr(v, "is_module", False)
    ]

    index_lines = [
        "# Python API Reference",
        "",
        "Auto-generated from qkrylov source via [Griffe](https://mkdocstrings.github.io/griffe/).",
        "",
        "| Module | Description |",
        "|--------|-------------|",
    ]

    for submod_name in sorted(submod_names):
        submod = mod.members[submod_name]
        doc_summary = safe_doc(submod).split("\n")[0] if safe_doc(submod) else f"The `qkrylov.{submod_name}` module."
        index_lines.append(f"| [`qkrylov.{submod_name}`](./{submod_name}/) | {doc_summary} |")

        # Build per-module page
        body: list[str] = [f"# `qkrylov.{submod_name}`", ""]
        submod_doc = safe_doc(submod)
        if submod_doc:
            body.append(submod_doc)
            body.append("")

        classes = []
        functions = []
        for m_name, member in submod.members.items():
            if m_name.startswith("_"):
                continue
            try:
                if getattr(member, "is_class", False):
                    classes.append((m_name, member))
                elif getattr(member, "is_function", False):
                    functions.append((m_name, member))
            except Exception:
                continue

        if classes:
            body.append("## Classes")
            body.append("")
            for cls_name, cls in sorted(classes, key=lambda x: x[0]):
                body.extend(render_class(cls_name, cls, prefix=f"qkrylov.{submod_name}"))

        if functions:
            body.append("## Functions")
            body.append("")
            for fn_name, fn in sorted(functions, key=lambda x: x[0]):
                body.extend(render_function(fn_name, fn, prefix=f"qkrylov.{submod_name}"))

        page_path = OUT_DIR / f"{submod_name}.md"
        write_page(page_path, f"qkrylov.{submod_name}", f"Python API reference for qkrylov.{submod_name}", body)
        print(f"[gen_python_api] Generated {page_path}")

    # Write index
    index_path = OUT_DIR / "index.md"
    write_page(index_path, "Python API Reference", "Auto-generated Python API reference for qkrylov", index_lines)
    print(f"[gen_python_api] Generated {index_path}")

else:
    # AST fallback: one page per .py file
    import ast
    py_dir = PY_BINDINGS / "qkrylov"
    index_lines = ["# Python API Reference", "", "| Module | Description |", "|--------|-------------|"]

    for py_file in sorted(py_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))
        mod_doc = ast.get_docstring(tree) or ""
        summary = mod_doc.split("\n")[0] if mod_doc else f"The {py_file.stem} module."
        index_lines.append(f"| [`qkrylov.{py_file.stem}`](./{py_file.stem}/) | {summary} |")

        body: list[str] = [f"# `qkrylov.{py_file.stem}`", ""]
        if mod_doc:
            body.append(mod_doc)
            body.append("")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                body.append(f"## `{node.name}`")
                body.append("")
                doc = ast.get_docstring(node) or ""
                if doc:
                    body.append(doc)
                    body.append("")
            elif isinstance(node, ast.FunctionDef):
                body.append(f"### `{node.name}`")
                body.append("")
                doc = ast.get_docstring(node) or ""
                if doc:
                    body.append(doc)
                    body.append("")

        page_path = OUT_DIR / f"{py_file.stem}.md"
        write_page(page_path, f"qkrylov.{py_file.stem}", f"Python API for {py_file.stem}", body)
        print(f"[gen_python_api] Generated {page_path}")

    index_path = OUT_DIR / "index.md"
    write_page(index_path, "Python API Reference", "Auto-generated Python API reference for qkrylov", index_lines)
    print(f"[gen_python_api] Generated {index_path}")

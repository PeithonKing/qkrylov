# Aritra's TODO — Repo Manager / Build System / DevOps

> **Your role:** Keep the build infrastructure pristine. Make Ananya's docs writing frictionless.
> The Polyglot Ghost Builder system is complete. Your job is the **plumbing**, not the prose.

---

## 🔴 Immediate (Do Now)

### 1. Resolve & Commit the Merge State
The upstream pull from Pritipriya's PR left us with a clean but uncommitted state.
- [x] Conflicts resolved (docs/index.md, docs/quickstart.md, all old MkDocs deletions)
- [x] Stage and commit the resolved merge. Suggested command:
  ```bash
  git add .
  git commit -m "chore: merge Pritipriya PR#42, port temp-docs to docs/api, resolve conflicts"
  git push
  ```
- Then push origin up to sync your fork.

### 2. Delete `docs_hooks.py` if it still exists
This was the old MkDocs Python hooks file. Zensical ignores it.
It is already in `.gitignore` but delete it from the working tree if present.
```bash
rm -f docs_hooks.py mkdocs.yml
```

### 3. Fix `.gitignore` for Ghost Files
Make sure these exact patterns are in `.gitignore`:
```
docs/getting_started/tutorial_hamiltonian/index.md
docs/getting_started/tutorial_hamiltonian/tutorial_hamiltonian.*
docs/getting_started/tutorial_hamiltonian/images/
```
The Ghost Builder generates these files at serve time and deletes them at exit.
If they are NOT in `.gitignore`, any contributor who accidentally runs `git add .`
during a live serve session will pollute the repo with generated files.

---

## 🟡 Short-Term (Before Ananya finishes writing)

### 4. Review Zensical Config Updates
**Every time Ananya creates a new `docs/getting_started/<tutorial_name>/` folder**,
she is instructed to wire it into `zensical.toml` herself (adding it to the `nav` and `[extra] polyglot_tutorials` arrays).

Your job is to **review her PRs** to make sure:
1. She didn't break the TOML syntax.
2. The paths match exactly what she named her folder.
That is literally all you need to do for new tutorials. The Ghost Builder auto-discovers them once they are in the config.

### 5. `.gitignore` Pattern for New Tutorials
For each new tutorial Ananya creates (e.g., `tutorial_ftlm`), add these to `.gitignore`:
```
docs/getting_started/tutorial_ftlm/index.md
docs/getting_started/tutorial_ftlm/tutorial_ftlm.*
docs/getting_started/tutorial_ftlm/images/
```

### 6. Wire `replications.md` into the Navbar
The file `docs/replications.md` already exists and has decent content (Bethe Ansatz 1D Heisenberg).
It is in `zensical.toml` under the nav. Test that it builds without broken links.
Fix any broken internal links that point to the deleted old MkDocs pages.

### 7. Fix Broken Links in `docs/theory/` Files
Pritipriya's `temp-docs` and the old subagent-generated theory pages have
broken links pointing to deleted pages (`../solvers/ftlm.md`, `../performance.md`, etc.)
When Ananya has finished writing the corresponding pages, update these links.
**Leave for now, fix after Ananya's writing is complete.**

Current broken link count from Zensical: ~58 warnings.

---

## 🟢 When Ananya Delivers Files

### 8. Vet Every File Ananya Delivers
The old subagent-generated stubs (`ecosystem.md`, `sciml_julia.md`, the old `api/` stubs)
used WRONG APIs (wrong package name `qkrylov` → `QuantumKrylov`, wrong function signatures).
Pritipriya's `api/python.md`, `api/julia.md`, `api/cpp.md`, `api/c-abi.md` are the ground truth.

**Before merging any Ananya file, run:**
```bash
python3 build_polyglot.py build
```
and check that the warning count does not go up significantly.

### 9. Verify Ananya's Ghost Builder Commits
For each new tutorial she writes (Jupyter notebook + optional `.jl` / `.cpp`):
- She drops files in `docs/getting_started/<tutorial_name>/main.ipynb` etc.
- She wires the path into `zensical.toml` (see step 4).
- The Ghost Builder auto-generates the download buttons, strips `# [ID:]` comments,
  and produces clean polyglot notebooks.

---

## 🔵 Infrastructure Backlog (When You Have Time)

### 10. Validate CI Pipeline
The `.github/workflows/deploy_docs.yaml` runs `python build_polyglot.py build_ci`.
Test that GH Actions still deploys successfully to `gh-pages` after the merge.
Pritipriya's PR updated the FTLM, Dynamics, and Lanczos solvers heavily —
if the Python bindings need to be recompiled for CI, the deploy might fail.

### 11. Consider Removing `temp-docs/` After Porting
The `temp-docs/` directory was Pritipriya's staging area. Now that its contents
have been moved to `docs/api/`, `temp-docs/` is empty and deleted.
Confirm it doesn't reappear in future PRs and add to `.gitignore` if needed:
```
temp-docs/
```

---

## Reference: Current Docs State

| File | Status |
|------|--------|
| `docs/index.md` | ✅ Good (Polyglot tabs, Zensical styled) |
| `docs/quickstart.md` | ✅ Good (Pritipriya updated, our stash wins) |
| `docs/benchmarks.md` | ✅ Complete (Interactive Plotly dashboard) |
| `docs/replications.md` | ⚠️ Decent, needs broken link fixes |
| `docs/contributing.md` | ❓ Needs review |
| `docs/api/python.md` | ✅ Pritipriya's comprehensive manual |
| `docs/api/julia.md` | ✅ Pritipriya's comprehensive manual |
| `docs/api/cpp.md` | ✅ Pritipriya's comprehensive manual |
| `docs/api/c-abi.md` | ✅ Pritipriya's comprehensive manual |
| `docs/api/index.md` | ❌ Stub, needs overview prose |
| `docs/getting_started/tutorial_hamiltonian/` | ✅ Complete (Ghost Builder, 3 download buttons) |
| `docs/getting_started/ecosystem.md` | ❌ Hallucinated, wrong API — give to Ananya |
| `docs/getting_started/sciml_julia.md` | ❌ Hallucinated, wrong API — give to Ananya |
| `docs/theory/matrix_free_spmv.md` | ⚠️ Subagent-written, needs vet |
| `docs/theory/lanczos_memory.md` | ⚠️ Subagent-written, needs vet |
| `docs/theory/precision_stability.md` | ⚠️ Subagent-written, needs vet |
| `docs/theory/exact_diag.md` | ⚠️ Has broken links |
| `docs/theory/krylov.md` | ⚠️ Has broken links |
| `docs/theory/spectral.md` | ⚠️ Has broken links |

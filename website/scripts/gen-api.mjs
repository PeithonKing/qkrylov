import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const WEBSITE_ROOT = path.resolve(__dirname, '..');      // website/
const REPO_ROOT = path.resolve(WEBSITE_ROOT, '..');      // repo root
const SCRIPTS_DIR = __dirname;                           // website/scripts/
const AUTO_DIR = path.resolve(REPO_ROOT, 'website/src/content/docs/api/auto');
const TOOLS_BIN = path.resolve(SCRIPTS_DIR, '.tools/bin');

console.log('[gen-api] Starting polyglot API documentation pipeline...');

// 1. Ensure target auto directories exist
fs.mkdirSync(path.join(AUTO_DIR, 'python'), { recursive: true });
fs.mkdirSync(path.join(AUTO_DIR, 'julia'), { recursive: true });
fs.mkdirSync(path.join(AUTO_DIR, 'cpp'), { recursive: true });

// 2. Ensure tools (Doxygen 1.18.0, Doxybook2 1.5.0)
const ensureToolsScript = path.join(SCRIPTS_DIR, 'ensure_tools.sh');
if (fs.existsSync(ensureToolsScript)) {
  console.log('[gen-api] Checking and bootstrapping tools via ensure_tools.sh...');
  const res = spawnSync('bash', [ensureToolsScript], { cwd: REPO_ROOT, stdio: 'inherit' });
  if (res.status !== 0) {
    console.warn(`[gen-api] Warning: ensure_tools.sh exited with status ${res.status}`);
  }
}

// Ensure tools directory is in PATH for child processes
if (fs.existsSync(TOOLS_BIN)) {
  process.env.PATH = `${TOOLS_BIN}:${process.env.PATH}`;
}

// 3. Generate Python API Docs via Griffe
console.log('[gen-api] Generating Python API documentation...');
const pyScript = path.join(SCRIPTS_DIR, 'gen_python_api.py');
let pyRes = spawnSync('uv', ['run', '--with', 'griffe', 'python', pyScript], { cwd: REPO_ROOT, stdio: 'inherit' });
if (pyRes.status !== 0) {
  console.log('[gen-api] uv invocation failed or not found, falling back to python3...');
  pyRes = spawnSync('python3', [pyScript], { cwd: REPO_ROOT, stdio: 'inherit' });
}
if (pyRes.status !== 0) {
  console.error('[gen-api] Fatal: Python API generation failed.');
  process.exit(1);
}

// 4. Generate Julia API Docs
console.log('[gen-api] Generating Julia API documentation...');
const jlScript = path.join(SCRIPTS_DIR, 'make_julia_docs.jl');
if (fs.existsSync(jlScript)) {
  const jlRes = spawnSync('julia', [jlScript], { cwd: REPO_ROOT, stdio: 'inherit' });
  if (jlRes.status !== 0) {
    console.error('[gen-api] Fatal: Julia API generation failed.');
    process.exit(1);
  }
}

// 5. Generate C++ API Docs via Doxygen and Doxybook2
console.log('[gen-api] Generating C++ API documentation...');
const doxyfile = path.join(SCRIPTS_DIR, 'Doxyfile');
const doxybookJson = path.join(SCRIPTS_DIR, 'doxybook.json');
const templatesDir = path.join(SCRIPTS_DIR, 'templates');
const xmlDir = path.join(SCRIPTS_DIR, '.tools/xml');
const cppOutDir = path.join(AUTO_DIR, 'cpp');

if (fs.existsSync(doxyfile)) {
  const doxygenBin = fs.existsSync(path.join(TOOLS_BIN, 'doxygen'))
    ? path.join(TOOLS_BIN, 'doxygen')
    : 'doxygen';
  const doxybook2Bin = fs.existsSync(path.join(TOOLS_BIN, 'doxybook2'))
    ? path.join(TOOLS_BIN, 'doxybook2')
    : 'doxybook2';

  // Run Doxygen (from SCRIPTS_DIR so relative paths resolve)
  execFileSync(doxygenBin, [doxyfile], { cwd: SCRIPTS_DIR, stdio: 'inherit' });

  // Run Doxybook2
  fs.mkdirSync(cppOutDir, { recursive: true });
  execFileSync(doxybook2Bin, [
    '-i', xmlDir,
    '-o', cppOutDir,
    '-c', doxybookJson,
    '-t', templatesDir,
  ], { cwd: SCRIPTS_DIR, stdio: 'inherit' });

  // Clean XML directory to avoid leaving rogue or unneeded files
  if (fs.existsSync(xmlDir)) {
    fs.rmSync(xmlDir, { recursive: true, force: true });
  }
}

// 6. Write Overview Index Page
const overviewPath = path.join(AUTO_DIR, 'index.md');
const overviewContent = `---
title: "API Reference Overview"
description: "qkrylov polyglot API reference for Python, Julia, and C++"
---

# qkrylov API Reference

Explore the high-performance matrix-free Krylov subspace methods API:

- [Python API Reference](/qkrylov/api/python/)
- [Julia API Reference](/qkrylov/api/julia/)
- [C++ Core API Reference](/qkrylov/api/cpp/)
`;
fs.writeFileSync(overviewPath, overviewContent, 'utf8');

console.log('[gen-api] Polyglot API documentation pipeline completed successfully.');

/**
 * Polyglot Jupyter Tutorial Vite Plugin for Astro Starlight.
 *
 * Intercepts .ipynb files during build, extracts code blocks from adjacent
 * main.cpp and main.jl matching # [ID: step] markers, decodes base64 images into
 * in-memory data URIs, formats cell outputs as comments, and emits in-memory MDX
 * with :::tabs directives.
 *
 * ZERO ghost files written to disk.
 */

import fs from 'node:fs';
import path from 'node:path';

export const ORACLE_PATTERNS = {
  // Matches Python cell ID marker at the beginning of source, with optional leading whitespace
  PYTHON_ID_MARKER: /^\s*#\s*\[ID:\s*([a-zA-Z0-9_]+)\s*\]\s*\r?\n?/,
  // Matches C++ block bounded by // [ID: <id>] and // [END: <id>]
  CPP_BLOCK: (id) => new RegExp(`//\\s*\\[ID:\\s*${id}\\s*\\]([\\s\\S]*?)//\\s*\\[END:\\s*${id}\\s*\\]`),
  // Matches Julia block bounded by # [ID: <id>] and # [END: <id>]
  JULIA_BLOCK: (id) => new RegExp(`#\\s*\\[ID:\\s*${id}\\s*\\]([\\s\\S]*?)#\\s*\\[END:\\s*${id}\\s*\\]`),
};

/**
 * Extract code block from C++ source using authoritative regex
 */
export function extractCppBlock(cppSource, blockId) {
  if (!cppSource) return `// Error: ID ${blockId} not found`;
  const match = cppSource.match(ORACLE_PATTERNS.CPP_BLOCK(blockId));
  if (!match) return `// Error: ID ${blockId} not found`;
  return match[1].trim();
}

/**
 * Extract code block from Julia source using authoritative regex
 */
export function extractJuliaBlock(jlSource, blockId) {
  if (!jlSource) return `# Error: ID ${blockId} not found`;
  const match = jlSource.match(ORACLE_PATTERNS.JULIA_BLOCK(blockId));
  if (!match) return `# Error: ID ${blockId} not found`;
  return match[1].trim();
}

/**
 * Format stream text output into commented python lines
 */
export function formatStreamOutput(textOutputs) {
  if (!textOutputs || textOutputs.length === 0) return '';
  const outStr = textOutputs.join('').trim();
  if (!outStr) return '';
  return '\n\n# Output:\n' + outStr.split(/\r?\n/).map(line => `# ${line}`).join('\n');
}

/**
 * Format base64 image data into in-memory data URI markdown
 */
export function formatBase64Image(base64Data, altText = 'Output Image') {
  const cleanBase64 = String(base64Data).replace(/\s+/g, '');
  return `\n![${altText}](data:image/png;base64,${cleanBase64})\n`;
}

/**
 * Parse notebook title from metadata or first H1 header
 */
export function parseNotebookTitle(nb, defaultTitle = 'Building the Hamiltonian') {
  if (nb.metadata?.title) return nb.metadata.title;
  for (const cell of nb.cells || []) {
    if (cell.cell_type === 'markdown') {
      const text = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
      const match = text.match(/^\s*#\s+([^\r\n]+)/m);
      if (match) return match[1].trim();
    }
  }
  return defaultTitle;
}

/**
 * Transform Jupyter notebook + C++ + Julia into Starlight MDX string purely in-memory
 */
export function transformPolyglotToStarlightMdx({
  notebook,
  cppSource = '',
  jlSource = '',
  includeFrontmatter = false,
  stripTitleHeading = false,
  title = null,
}) {
  const nb = typeof notebook === 'string' ? JSON.parse(notebook) : notebook;
  const mdOutput = [];
  const extractedImages = [];

  const effectiveTitle = title || parseNotebookTitle(nb);

  if (includeFrontmatter && effectiveTitle) {
    mdOutput.push(`---
title: "${effectiveTitle.replace(/"/g, '\\"')}"
---`);
  }

  let isFirstMarkdown = true;

  for (let cellIdx = 0; cellIdx < (nb.cells || []).length; cellIdx++) {
    const cell = nb.cells[cellIdx];

    if (cell.cell_type === 'markdown') {
      let text = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
      if (isFirstMarkdown && stripTitleHeading && effectiveTitle) {
        // Strip top-level # Title to prevent duplicate H1 when rendered with StarlightPage
        const escaped = effectiveTitle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        text = text.replace(new RegExp(`^\\s*#\\s+${escaped}\\s*\\r?\\n?`, 'm'), '');
        isFirstMarkdown = false;
      }
      if (text.trim()) {
        mdOutput.push(text);
      }
    } else if (cell.cell_type === 'code') {
      const source = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');

      const textOutputs = [];
      const imageMarkdowns = [];

      for (const out of cell.outputs || []) {
        if (out.output_type === 'stream') {
          const streamText = Array.isArray(out.text) ? out.text : [out.text || ''];
          textOutputs.push(...streamText);
        } else if (out.output_type === 'execute_result' || out.output_type === 'display_data') {
          const data = out.data || {};
          if (data['image/png']) {
            const rawB64 = Array.isArray(data['image/png']) ? data['image/png'].join('') : data['image/png'];
            const cleanB64 = String(rawB64).replace(/\s+/g, '');
            extractedImages.push({
              cellIndex: cellIdx,
              dataUri: `data:image/png;base64,${cleanB64}`,
            });
            imageMarkdowns.push(formatBase64Image(cleanB64));
          } else if (data['text/plain']) {
            const plainText = Array.isArray(data['text/plain']) ? data['text/plain'] : [data['text/plain'] || ''];
            textOutputs.push(...plainText);
          }
        }
      }

      const formattedOutput = formatStreamOutput(textOutputs);
      const markerMatch = source.match(ORACLE_PATTERNS.PYTHON_ID_MARKER);

      if (markerMatch) {
        const blockId = markerMatch[1];
        const cleanSource = source.slice(markerMatch[0].length);
        const pyCode = cleanSource + formattedOutput;
        const cppCode = extractCppBlock(cppSource, blockId);
        const jlCode = extractJuliaBlock(jlSource, blockId);

        const tabsBlock = [
          '::::tabs',
          ':::tab[Python]',
          '```python',
          pyCode.trim(),
          '```',
          ':::',
          ':::tab[C++]',
          '```cpp',
          cppCode.trim(),
          '```',
          ':::',
          ':::tab[Julia]',
          '```julia',
          jlCode.trim(),
          '```',
          ':::',
          '::::',
        ].join('\n');

        mdOutput.push(tabsBlock);
      } else {
        const pyCode = source + formattedOutput;
        mdOutput.push(`\`\`\`python\n${pyCode.trim()}\n\`\`\``);
      }

      if (imageMarkdowns.length > 0) {
        mdOutput.push(...imageMarkdowns);
      }
    }
  }

  return {
    mdx: mdOutput.join('\n\n'),
    images: extractedImages,
    ghostFilesWritten: 0,
  };
}

/**
 * Custom Vite plugin that intercepts .ipynb imports and provides in-memory MDX.
 */
export function polyglotIpynbPlugin(options = {}) {
  return {
    name: 'vite-plugin-polyglot-ipynb',
    enforce: 'pre',
    resolveId(id, importer) {
      if (id.endsWith('.ipynb')) {
        const resolved = importer
          ? path.resolve(path.dirname(importer), id)
          : path.resolve(id);
        return resolved + '.mdx';
      }
      if (id.endsWith('.ipynb.mdx')) {
        return id;
      }
    },
    load(id) {
      if (id.endsWith('.ipynb.mdx')) {
        const ipynbPath = id.slice(0, -4);
        if (!fs.existsSync(ipynbPath)) {
          return null;
        }

        if (typeof this.addWatchFile === 'function') {
          this.addWatchFile(ipynbPath);
        }

        const dir = path.dirname(ipynbPath);
        const baseName = path.basename(dir);
        const rawNb = fs.readFileSync(ipynbPath, 'utf8');
        const nb = JSON.parse(rawNb);

        // Find adjacent C++ source (main.cpp or <dir>.cpp)
        let cppSrc = '';
        const cppCandidates = [
          path.join(dir, 'main.cpp'),
          path.join(dir, `${baseName}.cpp`),
        ];
        for (const candidate of cppCandidates) {
          if (fs.existsSync(candidate)) {
            cppSrc = fs.readFileSync(candidate, 'utf8');
            if (typeof this.addWatchFile === 'function') {
              this.addWatchFile(candidate);
            }
            break;
          }
        }

        // Find adjacent Julia source (main.jl or <dir>.jl)
        let jlSrc = '';
        const jlCandidates = [
          path.join(dir, 'main.jl'),
          path.join(dir, `${baseName}.jl`),
        ];
        for (const candidate of jlCandidates) {
          if (fs.existsSync(candidate)) {
            jlSrc = fs.readFileSync(candidate, 'utf8');
            if (typeof this.addWatchFile === 'function') {
              this.addWatchFile(candidate);
            }
            break;
          }
        }

        const result = transformPolyglotToStarlightMdx({
          notebook: nb,
          cppSource: cppSrc,
          jlSource: jlSrc,
          includeFrontmatter: true,
          stripTitleHeading: true,
        });

        return {
          code: result.mdx,
          map: null,
        };
      }
    },
  };
}

export default polyglotIpynbPlugin;

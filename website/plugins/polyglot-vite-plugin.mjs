import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEBSITE_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DOWNLOADS_DIR = path.resolve(WEBSITE_ROOT, 'public', 'downloads');
const PUBLIC_IMAGES_DIR = path.resolve(WEBSITE_ROOT, 'public', '.generated_webp');

const ORACLE_PATTERNS = {
  PYTHON_ID_MARKER: /^#\s*\[ID:\s*(\w+)\]\s*\r?\n?/,
  CPP_BLOCK: (id) => new RegExp(`\\/\\/\\s*\\[ID:\\s*${id}\\]\\s*\\r?\\n([\\s\\S]*?)\\/\\/\\s*\\[END:\\s*${id}\\]`, 'm'),
  JULIA_BLOCK: (id) => new RegExp(`#\\s*\\[ID:\\s*${id}\\]\\s*\\r?\\n([\\s\\S]*?)#\\s*\\[END:\\s*${id}\\]`, 'm'),
};

export function extractCppBlock(cppSource, blockId) {
  if (!cppSource) return `// Error: ID ${blockId} not found`;
  const match = cppSource.match(ORACLE_PATTERNS.CPP_BLOCK(blockId));
  return match ? match[1].trim() : `// Error: ID ${blockId} not found`;
}

export function extractJuliaBlock(jlSource, blockId) {
  if (!jlSource) return `# Error: ID ${blockId} not found`;
  const match = jlSource.match(ORACLE_PATTERNS.JULIA_BLOCK(blockId));
  return match ? match[1].trim() : `# Error: ID ${blockId} not found`;
}

export function formatStreamOutput(textOutputs) {
  if (!textOutputs || textOutputs.length === 0) return '';
  const outStr = textOutputs.join('').trim();
  if (!outStr) return '';
  return '\n\n# Output:\n' + outStr.split(/\r?\n/).map((line) => `# ${line}`).join('\n');
}

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

export async function transformPolyglotToStarlightMdx({
  notebook,
  cppSource = '',
  jlSource = '',
  includeFrontmatter = false,
  stripTitleHeading = false,
  title = null,
  tutorialName = 'tutorial',
}) {
  const nb = typeof notebook === 'string' ? JSON.parse(notebook) : notebook;
  const mdOutput = [];
  
  // Ghost Script Buffers
  const cppAnnotated = ['/*\n * QKRYLOV FULLY ANNOTATED TUTORIAL\n */\n'];
  const jlAnnotated = ['#=\n = QKRYLOV FULLY ANNOTATED TUTORIAL\n =#\n'];
  let currentMdBuffer = [];

  const effectiveTitle = title || parseNotebookTitle(nb);

  // MDX import must come right after frontmatter, before any content
  const componentImport = `import PolyglotDownloadButtons from '@components/PolyglotDownloadButtons.astro';`;

  if (includeFrontmatter && effectiveTitle) {
    mdOutput.push(`---
title: "${effectiveTitle.replace(/"/g, '\\"')}"
---`);
  }

  // Import goes immediately after frontmatter
  mdOutput.push(componentImport);

  // JSX component call — single entry point, all 3 buttons live inside the component
  const buttonsJsx = `<PolyglotDownloadButtons tutorialName="${tutorialName}" />`;
  mdOutput.push(buttonsJsx);

  let isFirstMarkdown = true;
  let imageCounter = 0;
  
  // Clean notebook copy
  const cleanNb = JSON.parse(JSON.stringify(nb));

  for (let cellIdx = 0; cellIdx < (nb.cells || []).length; cellIdx++) {
    const cell = nb.cells[cellIdx];
    const cleanCell = cleanNb.cells[cellIdx];

    if (cell.cell_type === 'markdown') {
      let text = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
      if (isFirstMarkdown && stripTitleHeading && effectiveTitle) {
        const escaped = effectiveTitle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        text = text.replace(new RegExp(`^\\s*#\\s+${escaped}\\s*\\r?\\n?`, 'm'), '');
        isFirstMarkdown = false;
      }
      if (text.trim()) {
        mdOutput.push(text);
        currentMdBuffer.push(text);
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
            const buffer = Buffer.from(cleanB64, 'base64');
            
            // Convert to WebP on the fly
            const webpBuffer = await sharp(buffer).webp({ quality: 75 }).toBuffer();
            
            // Ensure public images directory exists
            const outDir = path.join(PUBLIC_IMAGES_DIR, tutorialName);
            fs.mkdirSync(outDir, { recursive: true });
            
            const filename = `plot_${cellIdx}_${imageCounter++}.webp`;
            fs.writeFileSync(path.join(outDir, filename), webpBuffer);
            
            imageMarkdowns.push(`\n![Output Plot](/qkrylov/.generated_webp/${tutorialName}/${filename})\n`);
          } else if (data['text/plain']) {
            const plainText = Array.isArray(data['text/plain']) ? data['text/plain'] : [data['text/plain'] || ''];
            textOutputs.push(...plainText);
          }
        }
      }

      const formattedOutput = formatStreamOutput(textOutputs);
      const markerMatch = source.match(ORACLE_PATTERNS.PYTHON_ID_MARKER);

      if (markerMatch) {
        // Clean notebook: remove ID tag
        if (Array.isArray(cleanCell.source)) {
          cleanCell.source = cleanCell.source.filter(line => !line.match(ORACLE_PATTERNS.PYTHON_ID_MARKER));
        } else {
          cleanCell.source = cleanCell.source.replace(ORACLE_PATTERNS.PYTHON_ID_MARKER, '');
        }

        const blockId = markerMatch[1];
        const cleanSource = source.slice(markerMatch[0].length);
        const pyCode = cleanSource + formattedOutput;
        const cppCode = extractCppBlock(cppSource, blockId);
        const jlCode = extractJuliaBlock(jlSource, blockId);

        // Inject markdown buffer into ghost scripts
        const mdText = currentMdBuffer.join('\n\n').trim();
        if (mdText) {
          cppAnnotated.push(`/*\n${mdText}\n*/\n`);
          jlAnnotated.push(`#=\n${mdText}\n=#\n`);
        }
        currentMdBuffer = []; // Reset buffer

        // Append actual code to ghost scripts
        cppAnnotated.push(cppCode + '\n');
        jlAnnotated.push(jlCode + '\n');

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

  // Ensure public downloads directory exists and write artifacts
  const outDir = path.join(PUBLIC_DOWNLOADS_DIR, tutorialName);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'main.cpp'), cppAnnotated.join('\n'));
  fs.writeFileSync(path.join(outDir, 'main.jl'), jlAnnotated.join('\n'));
  fs.writeFileSync(path.join(outDir, 'main.ipynb'), JSON.stringify(cleanNb, null, 1));

  return { mdx: mdOutput.join('\n\n') };
}

export function polyglotIpynbPlugin(options = {}) {
  return {
    name: 'vite-plugin-polyglot-ipynb',
    enforce: 'pre',
    resolveId(id, importer) {
      if (id.endsWith('.ipynb')) {
        const resolved = importer ? path.resolve(path.dirname(importer), id) : path.resolve(id);
        return resolved + '.mdx';
      }
      if (id.endsWith('.ipynb.mdx')) {
        return id;
      }
    },
    async load(id) {
      if (id.endsWith('.ipynb.mdx')) {
        const ipynbPath = id.slice(0, -4);
        if (!fs.existsSync(ipynbPath)) return null;

        if (typeof this.addWatchFile === 'function') this.addWatchFile(ipynbPath);

        const dir = path.dirname(ipynbPath);
        const baseName = path.basename(dir);
        const rawNb = fs.readFileSync(ipynbPath, 'utf8');
        
        let cppSrc = '';
        for (const candidate of [path.join(dir, 'main.cpp'), path.join(dir, `${baseName}.cpp`)]) {
          if (fs.existsSync(candidate)) {
            cppSrc = fs.readFileSync(candidate, 'utf8');
            if (typeof this.addWatchFile === 'function') this.addWatchFile(candidate);
            break;
          }
        }

        let jlSrc = '';
        for (const candidate of [path.join(dir, 'main.jl'), path.join(dir, `${baseName}.jl`)]) {
          if (fs.existsSync(candidate)) {
            jlSrc = fs.readFileSync(candidate, 'utf8');
            if (typeof this.addWatchFile === 'function') this.addWatchFile(candidate);
            break;
          }
        }

        const result = await transformPolyglotToStarlightMdx({
          notebook: rawNb,
          cppSource: cppSrc,
          jlSource: jlSrc,
          includeFrontmatter: true,
          stripTitleHeading: true,
          tutorialName: baseName
        });

        return { code: result.mdx, map: null };
      }
    },
  };
}

export default polyglotIpynbPlugin;

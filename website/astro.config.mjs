import { fileURLToPath } from 'url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mdx from '@astrojs/mdx';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';
import remarkTabs from './plugins/remark-tabs.mjs';
import polyglotIpynbPlugin from './plugins/polyglot-vite-plugin.mjs';

export default defineConfig({
  site: 'https://sjp95.github.io',
  base: process.env.ASTRO_BASE_PATH || '/qkrylov',
  integrations: [
    starlight({
      title: 'qkrylov',
      customCss: ['katex/dist/katex.min.css'],
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/sjp95/qkrylov' },
      ],
      sidebar: [
        { label: 'Overview', link: '/' },
        {
          label: 'Getting Started',
          items: [
            { label: 'Quickstart', link: '/quickstart/' },
            { label: 'Building the Hamiltonian', link: '/getting_started/tutorial_hamiltonian/' },
            { label: 'Ecosystem Interop (NumPy / SciPy / Torch)', link: '/getting_started/ecosystem/' },
            { label: 'SciML Workflow (Julia)', link: '/getting_started/sciml_julia/' },
          ],
        },
        {
          label: 'How-To Guides',
          items: [
            { label: 'Compute a Spectral Function', link: '/how-to/spectral_function/' },
            { label: 'Finite Temperature (FTLM)', link: '/how-to/finite_temperature/' },
            { label: 'Real-Time Dynamics', link: '/how-to/real_time_dynamics/' },
            { label: 'Choosing Precision & Backend', link: '/how-to/precision_and_backend/' },
          ],
        },
        {
          label: 'Theory',
          items: [
            { label: 'Overview', link: '/theory/' },
            { label: 'Exact Diagonalization (ED)', link: '/theory/exact_diag/' },
            { label: 'Krylov Subspace Methods', link: '/theory/krylov/' },
            { label: 'Matrix-Free SpMV on GPUs', link: '/theory/matrix_free_spmv/' },
            { label: 'Single-Pass vs Two-Pass Lanczos', link: '/theory/lanczos_memory/' },
            { label: 'Dual-Precision (FP32/FP64) Stability', link: '/theory/precision_stability/' },
            { label: 'Spectral Functions & Dynamics', link: '/theory/spectral/' },
          ],
        },
        {
          label: 'Benchmarks',
          collapsed: true,       // ← set false to expand by default
          items: [
            { label: 'Performance Results', link: '/benchmarks/' },
            { label: 'Replicated Papers', link: '/replications/' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            {
              label: 'Python',
              collapsed: true,   // ← set false to expand by default
              items: [{ autogenerate: { directory: 'api/auto/python' } }],
            },
            {
              label: 'Julia',
              collapsed: true,   // ← set false to expand by default
              items: [{ autogenerate: { directory: 'api/auto/julia' } }],
            },
            {
              label: 'C++',
              collapsed: true,   // ← set false to expand by default
              items: [{ autogenerate: { directory: 'api/auto/cpp' } }],
            },
          ],
        },
        { label: 'Contributing', link: '/contributing/' },
      ],
    }),
    mdx(),
  ],
  vite: {
    plugins: [polyglotIpynbPlugin()],
    resolve: {
      alias: {
        '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
      },
    },
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkDirective, remarkTabs, remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
});

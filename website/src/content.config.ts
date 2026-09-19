import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// docsLoader() reads from {srcDir}/content/docs/ by default.
// Since srcDir = '../docs/src' in astro.config.mjs,
// it resolves to docs/src/content/docs/ — where Ananya writes markdown.
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};

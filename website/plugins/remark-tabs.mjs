/**
 * Custom remark plugin to convert remark-directive :::tabs and :::tab[Label]
 * into Starlight <Tabs> and <TabItem> MDX elements (for .mdx) or native
 * <starlight-tabs> HTML HAST elements (for .md).
 */
export default function remarkTabs() {
  let globalTabsIndex = 0;

  function extractText(node) {
    if (!node) return '';
    if (typeof node.value === 'string') return node.value;
    if (Array.isArray(node.children)) return node.children.map(extractText).join('');
    return '';
  }

  function isColonDelimiter(node) {
    if (!node || node.type !== 'paragraph') return false;
    const text = extractText(node).trim();
    return text.length > 0 && /^(:{3,}\s*)+$/.test(text);
  }

  return (tree, file) => {
    const isMdx = Boolean(file?.path?.endsWith('.mdx') || file?.history?.[0]?.endsWith('.mdx'));

    // Phase 1: normalize flat :::tabs and :::tab siblings into nested hierarchy if needed
    const normalizeChildren = (parent) => {
      if (!parent.children) return;
      const newChildren = [];
      let currentTabs = null;

      for (const child of parent.children) {
        if (child.type === 'containerDirective' && child.name === 'tabs') {
          currentTabs = child;
          newChildren.push(child);
        } else if (child.type === 'containerDirective' && child.name === 'tab') {
          if (currentTabs && !currentTabs.children.includes(child)) {
            currentTabs.children.push(child);
          } else {
            newChildren.push(child);
          }
        } else {
          if (isColonDelimiter(child)) {
            continue;
          }
          currentTabs = null;
          newChildren.push(child);
        }
      }
      parent.children = newChildren;
      for (const child of parent.children) normalizeChildren(child);
    };

    normalizeChildren(tree);

    let hasTabs = false;

    // Phase 2: transform containerDirective nodes
    function walk(node) {
      if (!node) return;
      if (node.type === 'containerDirective') {
        if (node.name === 'tabs') {
          hasTabs = true;
          const syncKey = node.attributes?.syncKey || 'polyglot';

          if (isMdx) {
            node.type = 'mdxJsxFlowElement';
            node.name = 'Tabs';
            node.attributes = [{ type: 'mdxJsxAttribute', name: 'syncKey', value: syncKey }];
            node.children = (node.children || []).filter(
              c => c.type === 'containerDirective' && c.name === 'tab'
            );
          } else {
            const tabsInstanceId = globalTabsIndex++;
            node.data = {
              ...(node.data || {}),
              hName: 'starlight-tabs',
              hProperties: { 'data-sync-key': syncKey },
            };

            // Collect tabs info
            const tabChildren = (node.children || []).filter(
              c => c.type === 'containerDirective' && c.name === 'tab'
            );

            const tabsMeta = tabChildren.map((tabNode, i) => {
              let label = tabNode.attributes?.label;
              const labelNode = tabNode.children?.find(c => c.data?.directiveLabel);
              if (labelNode) {
                const fullLabel = extractText(labelNode).trim();
                if (fullLabel) label = fullLabel;
                tabNode.children = tabNode.children.filter(c => c !== labelNode);
              }
              if (!label) label = `Tab ${i + 1}`;
              const panelId = `tab-panel-${tabsInstanceId}-${i}`;
              const tabId = `tab-${tabsInstanceId}-${i}`;
              return { label, panelId, tabId, node: tabNode };
            });

            // Format tabpanel children
            tabsMeta.forEach(({ label, panelId, tabId, node: tabNode }, i) => {
              const hProps = {
                role: 'tabpanel',
                id: panelId,
                'aria-labelledby': tabId,
                'data-label': label,
              };
              if (i > 0) hProps.hidden = true;
              tabNode.data = {
                ...(tabNode.data || {}),
                hName: 'div',
                hProperties: hProps,
              };
            });

            // Construct tablist header
            const tablistWrapper = {
              type: 'parent',
              data: {
                hName: 'div',
                hProperties: { class: 'tablist-wrapper not-content' },
              },
              children: [
                {
                  type: 'parent',
                  data: {
                    hName: 'ul',
                    hProperties: { role: 'tablist' },
                  },
                  children: tabsMeta.map(({ label, panelId, tabId }, i) => ({
                    type: 'parent',
                    data: {
                      hName: 'li',
                      hProperties: { role: 'presentation', class: 'tab' },
                    },
                    children: [
                      {
                        type: 'link',
                        url: `#${panelId}`,
                        data: {
                          hName: 'a',
                          hProperties: {
                            role: 'tab',
                            href: `#${panelId}`,
                            id: tabId,
                            'aria-selected': i === 0 ? 'true' : 'false',
                            tabindex: i === 0 ? 0 : -1,
                          },
                        },
                        children: [{ type: 'text', value: label }],
                      },
                    ],
                  })),
                },
              ],
            };

            // Prepend tablist header
            node.children = [tablistWrapper, ...tabChildren];
          }
        } else if (node.name === 'tab') {
          hasTabs = true;
          let label = node.attributes?.label;
          const labelNode = node.children?.find(c => c.data?.directiveLabel);
          if (labelNode) {
            const fullLabel = extractText(labelNode).trim();
            if (fullLabel) label = fullLabel;
            node.children = node.children.filter(c => c !== labelNode);
          }
          if (!label) label = 'Tab';

          if (isMdx) {
            node.type = 'mdxJsxFlowElement';
            node.name = 'TabItem';
            node.attributes = [{ type: 'mdxJsxAttribute', name: 'label', value: label }];
          }
        }
      }
      if (node.children && Array.isArray(node.children)) {
        for (const child of node.children) walk(child);
      }
    }
    walk(tree);

    // Phase 3: If tabs were found in MDX, ensure Tabs and TabItem are imported
    if (isMdx && hasTabs && tree.children) {
      const hasImport = tree.children.some(
        child => child.type === 'mdxjsEsm' && child.value && child.value.includes('@astrojs/starlight/components')
      );
      if (!hasImport) {
        tree.children.unshift({
          type: 'mdxjsEsm',
          value: "import { Tabs, TabItem } from '@astrojs/starlight/components';",
          data: {
            estree: {
              type: 'Program',
              sourceType: 'module',
              body: [
                {
                  type: 'ImportDeclaration',
                  specifiers: [
                    {
                      type: 'ImportSpecifier',
                      imported: { type: 'Identifier', name: 'Tabs' },
                      local: { type: 'Identifier', name: 'Tabs' },
                    },
                    {
                      type: 'ImportSpecifier',
                      imported: { type: 'Identifier', name: 'TabItem' },
                      local: { type: 'Identifier', name: 'TabItem' },
                    },
                  ],
                  source: {
                    type: 'Literal',
                    value: '@astrojs/starlight/components',
                    raw: "'@astrojs/starlight/components'",
                  },
                },
              ],
            },
          },
        });
      }
    }
  };
}

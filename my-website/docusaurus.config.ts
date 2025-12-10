import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'A comprehensive guide to modern robotics with ROS 2, Simulation, AI, and Vision-Language-Action models',
  favicon: 'img/robotics-favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://physical-ai-robotics-textbook.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'robotics-textbook', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotics', // Usually your repo name.

  onBrokenLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'], // Adding Urdu for multilingual support
  },

  themes: [
    // Add any custom themes here
  ],
  plugins: [
    async function progressTrackerPlugin(context, options) {
      return {
        name: 'progress-tracker-plugin',
        async loadContent() {
          // Nothing to load
        },
        async contentLoaded({content, actions}) {
          // Nothing to do during content loading
        },
        injectHtmlTags() {
          return {
            headTags: [
              {
                tagName: 'script',
                innerHTML: `
                  // Initialize the progress tracking when the page loads
                  document.addEventListener('DOMContentLoaded', () => {
                    // Initialize the progress tracking object if it doesn't exist
                    if (!window.RoboticsTextbookProgress) {
                      const completedChapters = JSON.parse(localStorage.getItem('robotics-textbook-progress')) || [];

                      window.RoboticsTextbookProgress = {
                        completedChapters,
                        markChapterComplete: (chapterId) => {
                          if (!window.RoboticsTextbookProgress.completedChapters.includes(chapterId)) {
                            window.RoboticsTextbookProgress.completedChapters.push(chapterId);
                            localStorage.setItem(
                              'robotics-textbook-progress',
                              JSON.stringify(window.RoboticsTextbookProgress.completedChapters)
                            );
                          }
                        },
                        markChapterIncomplete: (chapterId) => {
                          window.RoboticsTextbookProgress.completedChapters =
                            window.RoboticsTextbookProgress.completedChapters.filter(id => id !== chapterId);
                          localStorage.setItem(
                            'robotics-textbook-progress',
                            JSON.stringify(window.RoboticsTextbookProgress.completedChapters)
                          );
                        },
                        isChapterComplete: (chapterId) => {
                          return window.RoboticsTextbookProgress.completedChapters.includes(chapterId);
                        },
                        getProgressPercentage: (moduleChapters) => {
                          const completedInModule = moduleChapters.filter(chapterId =>
                            window.RoboticsTextbookProgress.completedChapters.includes(chapterId)
                          ).length;
                          return Math.round((completedInModule / moduleChapters.length) * 100);
                        }
                      };
                    }
                  });
                `,
              },
            ],
          };
        },
      };
    },
    // Plugin to initialize RAG service
    async function ragServicePlugin(context, options) {
      return {
        name: 'rag-service-plugin',
        async loadContent() {
          // Nothing to load
        },
        async contentLoaded({content, actions}) {
          // Nothing to do during content loading
        },
        injectHtmlTags() {
          return {
            headTags: [
              {
                tagName: 'script',
                innerHTML: `
                  // RAG Service implementation
                  class RagService {
                    constructor() {
                      this.chunkSize = 1024;  // As specified in the requirements
                      this.overlap = 200;     // As specified in the requirements
                      this.chunks = [];
                      this.documents = new Map();
                    }

                    /**
                     * Indexes content for retrieval
                     * @param {string} docId - Document identifier
                     * @param {string} content - Content to index
                     */
                    indexDocument(docId, content) {
                      // Add document to our collection
                      this.documents.set(docId, content);

                      // Create chunks with specified parameters
                      const chunks = this.createChunks(content, docId);

                      // Add chunks to our collection
                      chunks.forEach((chunk, index) => {
                        this.chunks.push({
                          id: \`\${docId}-chunk-\${index}\`,
                          docId,
                          content: chunk.text,
                          start: chunk.start,
                          end: chunk.end,
                          embedding: this.createEmbedding(chunk.text) // In a real implementation, this would create actual embeddings
                        });
                      });

                      console.log(\`Indexed document \${docId} with \${chunks.length} chunks\`);
                    }

                    /**
                     * Creates chunks with specified chunk_size and overlap
                     * @param {string} content - Content to chunk
                     * @param {string} docId - Document ID for the chunks
                     * @returns {Array} - Array of chunks with text, start, and end positions
                     */
                    createChunks(content, docId) {
                      const chunks = [];
                      let start = 0;

                      while (start < content.length) {
                        // Determine the end position for this chunk
                        let end = start + this.chunkSize;

                        // If this is not the last chunk, create an overlap
                        if (end < content.length) {
                          // Find a good breaking point near the end to avoid cutting sentences
                          const breakpoint = this.findBreakpoint(content, end, end - this.overlap);
                          end = breakpoint;
                        } else {
                          end = content.length;
                        }

                        // Extract the chunk text
                        const text = content.substring(start, end);

                        // Add the chunk to our array
                        chunks.push({
                          text,
                          start,
                          end,
                          docId
                        });

                        // Move the start position, accounting for overlap
                        start = end - this.overlap;

                        // If start is >= content length, we're done
                        if (start >= content.length) {
                          break;
                        }

                        // Prevent infinite loops in case of unexpected conditions
                        if (start >= end) {
                          start = end + 1;
                        }
                      }

                      return chunks;
                    }

                    /**
                     * Finds a good breakpoint for chunking (e.g., at sentence or paragraph boundaries)
                     * @param {string} content - Content to find breakpoint in
                     * @param {number} fallbackEnd - Fallback end position
                     * @param {number} searchStart - Starting position for the search
                     * @returns {number} - Position to break at
                     */
                    findBreakpoint(content, fallbackEnd, searchStart) {
                      // Look for a good breaking point near the end of the chunk
                      const searchEnd = Math.min(fallbackEnd + 50, content.length); // Look ahead a bit
                      const slice = content.slice(searchStart, searchEnd);

                      // Look for sentence boundaries first
                      let lastSentence = slice.lastIndexOf('. ');
                      if (lastSentence !== -1) {
                        return searchStart + lastSentence + 2; // Include the period and space
                      }

                      // Then look for paragraph breaks
                      let lastPara = slice.lastIndexOf('\\n\\n');
                      if (lastPara !== -1) {
                        return searchStart + lastPara + 2; // Include both newlines
                      }

                      // Then look for sentence end without space
                      let lastSentenceAlt = slice.lastIndexOf('.');
                      if (lastSentenceAlt !== -1) {
                        return searchStart + lastSentenceAlt + 1;
                      }

                      // Finally, just use the fallback if no good breakpoints found
                      return fallbackEnd;
                    }

                    /**
                     * Creates a simple embedding representation of text
                     * In a real implementation, this would use a proper embedding model
                     * @param {string} text - Text to create embedding for
                     * @returns {Array} - Embedding array
                     */
                    createEmbedding(text) {
                      // This is a simplified representation - in a real system this would be
                      // a high-dimensional vector from a model like SentenceTransformers
                      return Array.from({ length: 384 }, (_, i) => {
                        // Create a pseudo-embedding based on character codes
                        let sum = 0;
                        for (let j = 0; j < text.length; j++) {
                          sum += text.charCodeAt(j) * (j + 1);
                        }
                        return Math.sin(sum * (i + 1) / text.length);
                      });
                    }

                    /**
                     * Searches for relevant chunks based on a query
                     * @param {string} query - Query text
                     * @returns {Array} - Array of relevant chunks with scores
                     */
                    search(query, maxResults = 5) {
                      // Create embedding for the query
                      const queryEmbedding = this.createEmbedding(query);

                      // Calculate similarity scores for all chunks
                      const scoredChunks = this.chunks.map(chunk => {
                        const similarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
                        return {
                          ...chunk,
                          similarity
                        };
                      });

                      // Sort by similarity score (descending) and return top results
                      return scoredChunks
                        .sort((a, b) => b.similarity - a.similarity)
                        .slice(0, maxResults);
                    }

                    /**
                     * Calculates cosine similarity between two vectors
                     * @param {Array} vecA - First vector
                     * @param {Array} vecB - Second vector
                     * @returns {number} - Cosine similarity score
                     */
                    cosineSimilarity(vecA, vecB) {
                      if (vecA.length !== vecB.length) {
                        throw new Error("Vectors must have the same length");
                      }

                      let dotProduct = 0;
                      let magnitudeA = 0;
                      let magnitudeB = 0;

                      for (let i = 0; i < vecA.length; i++) {
                        dotProduct += vecA[i] * vecB[i];
                        magnitudeA += vecA[i] ** 2;
                        magnitudeB += vecB[i] ** 2;
                      }

                      magnitudeA = Math.sqrt(magnitudeA);
                      magnitudeB = Math.sqrt(magnitudeB);

                      if (magnitudeA === 0 || magnitudeB === 0) {
                        return 0; // If either vector is zero, similarity is 0
                      }

                      return dotProduct / (magnitudeA * magnitudeB);
                    }

                    /**
                     * Gets the source document for a chunk
                     * @param {string} chunkId - Chunk identifier
                     * @returns {Object} - Document information
                     */
                    getChunkSource(chunkId) {
                      const chunk = this.chunks.find(c => c.id === chunkId);
                      if (!chunk) return null;

                      return {
                        id: chunk.docId,
                        content: this.documents.get(chunk.docId),
                        chunkText: chunk.content
                      };
                    }
                  }

                  // Create a singleton instance
                  const ragService = new RagService();

                  // Index all textbook content when the service is initialized
                  // In a real implementation, this would happen when content is added
                  window.RagService = ragService;

                  // Initialize the RAG service when the page loads
                  document.addEventListener('DOMContentLoaded', () => {
                    // The RagService is now available via window.RagService
                  });
                `,
              },
            ],
          };
        },
      };
    },
  ],
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/robotics-textbook/physical-ai-humanoid-robotics/edit/main/',
          routeBasePath: '/', // Serve docs at the root route
          beforeDefaultRemarkPlugins: [
            [() => {
              return (tree) => {
                // This is a placeholder for the breadcrumbs plugin
                // In practice, you'd manipulate the MDX AST to add breadcrumbs at the top
              };
            }, {}],
          ],
        },
        blog: false, // Disable blog since we're focusing on textbook content
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/robotics-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    // Accessibility improvements
    metadata: [
      { name: 'keywords', content: 'robotics, AI, textbook, education, STEM' },
      { name: 'author', content: 'Physical AI & Humanoid Robotics Textbook Team' },
      { name: 'robots', content: 'index, follow' },
      { name: 'theme-color', content: '#2196F3' },
      { property: 'og:title', content: 'Physical AI & Humanoid Robotics Textbook' },
      { property: 'og:description', content: 'A comprehensive guide to modern robotics with ROS 2, Simulation, AI, and Vision-Language-Action models' },
      { property: 'og:type', content: 'website' },
      { property: 'og:url', content: 'https://physical-ai-robotics-textbook.com' },
    ],
    navbar: {
      title: 'Physical AI & Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Textbook Logo',
        src: 'img/robotics-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbook',
          position: 'left',
          label: 'Textbook',
        },
        {
          to: '/personalization-dashboard',
          label: 'Dashboard',
          position: 'right',
        },
        {
          href: 'https://github.com/robotics-textbook/physical-ai-humanoid-robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/intro',
            },
            {
              label: 'Module 1: ROS 2',
              to: '/modules/module-1-ros',
            },
            {
              label: 'Module 2: Simulation',
              to: '/modules/module-2-simulation',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Personalization Dashboard',
              to: '/personalization-dashboard',
            },
            {
              label: 'Textbook Summary',
              to: '/textbook-summary',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/robotics-textbook/physical-ai-humanoid-robotics',
            },
            {
              label: 'Community',
              href: 'https://discordapp.com/invite/docusaurus',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

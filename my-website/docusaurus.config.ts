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
    locales: ['en'], // Keeping only English to remove language selector
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
    // Removed the old rag service plugin since we're now using the rag-service.js file
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
      { name: 'viewport', content: 'width=device-width, initial-scale=1.0' },
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
        src: 'img/3765210.webp',
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

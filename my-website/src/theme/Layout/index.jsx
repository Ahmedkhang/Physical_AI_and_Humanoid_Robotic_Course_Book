// src/theme/Layout/index.jsx
import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { useLocation } from '@docusaurus/router';
import Chatbot from '@site/src/components/Chatbot';
import AccessibilityToolbar from '@site/src/components/AccessibilityToolbar';
import { UserProfileProvider } from '@site/src/contexts/UserProfileContext';
import { LanguageProvider } from '@site/src/contexts/LanguageContext';

// Define the breadcrumbs mapping
const BREADCRUMB_MAP = {
  '/': [{ label: 'Home', href: '/' }],
  '/modules/module-1-ros': [
    { label: 'Home', href: '/' },
    { label: 'Module 1: The Robotic Nervous System (ROS 2)', href: '/modules/module-1-ros' }
  ],
  '/modules/module-2-simulation': [
    { label: 'Home', href: '/' },
    { label: 'Module 2: The Digital Twin (Gazebo & Unity)', href: '/modules/module-2-simulation' }
  ],
  '/modules/module-3-ai': [
    { label: 'Home', href: '/' },
    { label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)', href: '/modules/module-3-ai' }
  ],
  '/modules/module-4-vla': [
    { label: 'Home', href: '/' },
    { label: 'Module 4: Vision-Language-Action (VLA)', href: '/modules/module-4-vla' }
  ]
};

// Function to generate breadcrumbs based on current path
const generateBreadcrumbs = (pathname) => {
  if (pathname.startsWith('/docs/modules/module-1-ros/chapter-')) {
    return [
      { label: 'Home', href: '/' },
      { label: 'Module 1: The Robotic Nervous System (ROS 2)', href: '/modules/module-1-ros' },
      { label: pathname.split('/')[3].replace(/-/g, ' '), href: pathname }
    ];
  } else if (pathname.startsWith('/docs/modules/module-2-simulation/chapter-')) {
    return [
      { label: 'Home', href: '/' },
      { label: 'Module 2: The Digital Twin (Gazebo & Unity)', href: '/modules/module-2-simulation' },
      { label: pathname.split('/')[3].replace(/-/g, ' '), href: pathname }
    ];
  } else if (pathname.startsWith('/docs/modules/module-3-ai/chapter-')) {
    return [
      { label: 'Home', href: '/' },
      { label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)', href: '/modules/module-3-ai' },
      { label: pathname.split('/')[3].replace(/-/g, ' '), href: pathname }
    ];
  } else if (pathname.startsWith('/docs/modules/module-4-vla/chapter-')) {
    return [
      { label: 'Home', href: '/' },
      { label: 'Module 4: Vision-Language-Action (VLA)', href: '/modules/module-4-vla' },
      { label: pathname.split('/')[3].replace(/-/g, ' '), href: pathname }
    ];
  }

  // Default breadcrumbs for module overview pages
  return BREADCRUMB_MAP[pathname] || [{ label: 'Home', href: '/' }];
};

const Layout = (props) => {
  const { pathname } = useLocation();
  const breadcrumbs = generateBreadcrumbs(pathname);

  return (
    <LanguageProvider>
      <UserProfileProvider>
        <>
          <AccessibilityToolbar />
          <OriginalLayout {...props}>
            <div className="breadcrumbs-container">
              <nav className="breadcrumbs" aria-label="breadcrumbs">
                {breadcrumbs.map((item, index) => (
                  <React.Fragment key={item.href}>
                    {index > 0 && <span className="breadcrumb-separator"> › </span>}
                    <a
                      className={`breadcrumb-item ${index === breadcrumbs.length - 1 ? 'active' : ''}`}
                      href={item.href}
                    >
                      {item.label}
                    </a>
                  </React.Fragment>
                ))}
              </nav>
            </div>
            <div id="main-content">
              {props.children}
            </div>
          </OriginalLayout>
          <Chatbot />
          <style jsx>{`
            .breadcrumbs-container {
              padding: 0.5rem 0;
              margin-bottom: 1rem;
              border-bottom: 1px solid #eee;
            }

            .breadcrumbs {
              padding: 0 1rem;
              font-size: 0.85rem;
              color: #666;
            }

            .breadcrumb-item {
              color: #666;
              text-decoration: none;
            }

            .breadcrumb-item:hover {
              text-decoration: underline;
            }

            .breadcrumb-item.active {
              color: #333;
              font-weight: bold;
              pointer-events: none;
            }

            .breadcrumb-separator {
              margin: 0 0.5rem;
              color: #999;
            }

            #main-content {
              min-height: calc(100vh - 200px); /* Adjust based on header height */
            }
          `}</style>
        </>
      </UserProfileProvider>
    </LanguageProvider>
  );
};

export default Layout;
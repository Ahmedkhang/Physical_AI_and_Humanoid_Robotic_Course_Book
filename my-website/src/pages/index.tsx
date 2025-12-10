import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero hero--primary">
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className="hero-buttons">
          <Link
            className="button button--secondary button--lg"
            to="/intro">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

// Module card component
function ModuleCard({ title, description, link, color }: { title: string, description: string, link: string, color: string }) {
  return (
    <div className="col col--3" style={{ padding: '10px' }}>
      <Link to={link} className="card" style={{
        backgroundColor: '#2d2d2d',
        color: 'white',
        padding: '2rem',
        borderRadius: '8px',
        minHeight: '250px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        transition: 'transform 0.3s ease, box-shadow 0.3s ease',
        border: '1px solid #444',
        height: '100%'
      }}>
        <div>
          <div style={{
            width: '50px',
            height: '50px',
            backgroundColor: color,
            borderRadius: '8px',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.5rem'
          }}>
            {title.charAt(0)}
          </div>
          <Heading as="h3" style={{ color: 'white' }}>{title}</Heading>
          <p style={{ color: '#ccc', flex: 1 }}>{description}</p>
        </div>
        <div style={{ marginTop: '1rem', color: '#4da6ff', fontWeight: 'bold' }}>
          Explore →
        </div>
      </Link>
    </div>
  );
}

// Modules section
function ModulesSection() {
  const modules = [
    {
      id: 'module-1-ros',
      title: 'Module 1: The Robotic Nervous System (ROS 2)',
      description: 'Foundational concepts of ROS 2 architecture, nodes, communication primitives, topics, services, actions, packages, and debugging tools.',
      link: '/modules/module-1-ros',
      color: '#4CAF50'
    },
    {
      id: 'module-2-simulation',
      title: 'Module 2: The Digital Twin (Gazebo & Unity)',
      description: 'Simulation environments for robotics, including Gazebo, Unity integration, robot modeling, physics engines, and sim-to-real transfer.',
      link: '/modules/module-2-simulation',
      color: '#2196F3'
    },
    {
      id: 'module-3-ai',
      title: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      description: 'AI and machine learning applications in robotics, including perception, planning, learning, navigation, manipulation, and decision-making.',
      link: '/modules/module-3-ai',
      color: '#FF9800'
    },
    {
      id: 'module-4-vla',
      title: 'Module 4: Vision-Language-Action (VLA)',
      description: 'Cutting-edge embodied AI with Vision-Language-Action models, reasoning, human-robot interaction, and advanced intelligence concepts.',
      link: '/modules/module-4-vla',
      color: '#9C27B0'
    }
  ];

  return (
    <section style={{ padding: '4rem 0', backgroundColor: '#1a1a1a' }}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" style={{
              textAlign: 'center',
              color: 'white',
              marginBottom: '3rem',
              fontSize: '2rem'
            }}>
              Course Modules
            </Heading>
          </div>
        </div>
        <div className="row">
          {modules.map((module, index) => (
            <ModuleCard
              key={index}
              title={module.title}
              description={module.description}
              link={module.link}
              color={module.color}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Textbook - A comprehensive guide to modern robotics">
      <HomepageHeader />
      <main>
        <ModulesSection />
      </main>
      <style jsx>{`
        .hero-buttons {
          margin-top: 2rem;
          display: flex;
          gap: 1rem;
          justify-content: center;
        }
      `}</style>
    </Layout>
  );
}

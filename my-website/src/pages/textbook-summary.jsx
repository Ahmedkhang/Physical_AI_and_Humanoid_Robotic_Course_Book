// src/pages/textbook-summary.jsx
import React from 'react';
import Layout from '@theme/Layout';
import { useUserProfile } from '@site/src/contexts/UserProfileContext';

// Safe hook that provides fallback data during server-side rendering
const useUserProfileSafe = () => {
  try {
    return useUserProfile();
  } catch (error) {
    // During static generation, context might not be available
    // Return default values
    return {
      profile: {
        completedChapters: [],
        preferences: {
          darkMode: false,
          fontSize: 'medium',
          notifications: true,
        }
      },
      loading: false
    };
  }
};

const TextbookSummary = () => {
  const { profile } = useUserProfileSafe();

  // Calculate overall progress
  const totalChapters = 27;
  const completedChapters = profile?.completedChapters?.length || 0;
  const progressPercentage = Math.round((completedChapters / totalChapters) * 100);

  // Module breakdown
  const modules = [
    { id: 'module-1-ros', name: 'Module 1: The Robotic Nervous System (ROS 2)', chapters: 7 },
    { id: 'module-2-simulation', name: 'Module 2: The Digital Twin (Gazebo & Unity)', chapters: 6 },
    { id: 'module-3-ai', name: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)', chapters: 8 },
    { id: 'module-4-vla', name: 'Module 4: Vision-Language-Action (VLA)', chapters: 6 }
  ];
  
  // Calculate progress per module
  const moduleProgress = modules.map(module => {
    const moduleChapters = Array.from({length: module.chapters}, (_, i) => 
      `${module.id}/chapter-${i === 0 && module.id === 'module-1-ros' ? 1 : (module.id === 'module-1-ros' ? i+1 : module.id === 'module-2-simulation' ? i+8 : module.id === 'module-3-ai' ? i+14 : i+22)}`
    );
    
    const completedInModule = profile.completedChapters.filter(chapterId => 
      moduleChapters.some(mc => chapterId.includes(mc.split('/')[1]))
    ).length;
    
    return {
      ...module,
      completed: completedInModule,
      percentage: Math.round((completedInModule / module.chapters) * 100)
    };
  });
  
  return (
    <Layout title="Textbook Summary" description="Overview of the Physical AI & Humanoid Robotics Textbook">
      <div className="textbook-summary-container">
        <div className="summary-header">
          <h1>Physical AI & Humanoid Robotics Textbook</h1>
          <p>Comprehensive Guide to Modern Robotics with ROS 2, Simulation, AI, and Vision-Language-Action Models</p>
        </div>
        
        <div className="summary-overview">
          <div className="overall-progress">
            <h2>Learning Progress</h2>
            <div className="progress-circle">
              <svg viewBox="0 0 36 36">
                <path
                  d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#eee"
                  strokeWidth="3"
                />
                <path
                  d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#2196F3"
                  strokeWidth="3"
                  strokeDasharray={`${progressPercentage}, 100`}
                />
              </svg>
              <div className="progress-text">
                <span className="percentage">{progressPercentage}%</span>
                <span className="count">{completedChapters}/{totalChapters} chapters</span>
              </div>
            </div>
          </div>
          
          <div className="progress-breakdown">
            <h2>Module Progress</h2>
            {moduleProgress.map((module, index) => (
              <div key={index} className="module-progress">
                <div className="module-header">
                  <h3>{module.name}</h3>
                  <span>{module.completed}/{module.chapters} ({module.percentage}%)</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${module.percentage}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="summary-content">
          <h2>Textbook Overview</h2>
          <p>
            This comprehensive textbook covers four essential modules of modern robotics, from foundational 
            concepts in ROS 2 to cutting-edge Vision-Language-Action models. The curriculum is designed 
            following pedagogical best practices with measurable learning outcomes, hands-on labs, 
            and practical applications.
          </p>
          
          <div className="modules-grid">
            <div className="module-card">
              <h3>Module 1: The Robotic Nervous System (ROS 2)</h3>
              <p>
                Foundational concepts of ROS 2 architecture, nodes, communication primitives, 
                topics, services, actions, parameters, packages, and debugging tools.
              </p>
              <ul>
                <li>7 chapters with hands-on labs</li>
                <li>Comprehensive coverage of middleware concepts</li>
                <li>Practical examples and MCQs</li>
              </ul>
            </div>
            
            <div className="module-card">
              <h3>Module 2: The Digital Twin (Gazebo & Unity)</h3>
              <p>
                Simulation environments for robotics, including Gazebo, Unity integration, 
                robot modeling, physics engines, and sim-to-real transfer.
              </p>
              <ul>
                <li>6 chapters with practical exercises</li>
                <li>Simulation best practices</li>
                <li>Bridge between simulation and reality</li>
              </ul>
            </div>
            
            <div className="module-card">
              <h3>Module 3: The AI-Robot Brain (NVIDIA Isaac)</h3>
              <p>
                AI and machine learning applications in robotics, including perception, 
                planning, learning, navigation, manipulation, and decision-making.
              </p>
              <ul>
                <li>8 chapters covering advanced topics</li>
                <li>Perception and action systems</li>
                <li>Multi-robot coordination</li>
              </ul>
            </div>
            
            <div className="module-card">
              <h3>Module 4: Vision-Language-Action (VLA)</h3>
              <p>
                Cutting-edge embodied AI with Vision-Language-Action models, reasoning, 
                human-robot interaction, and advanced intelligence concepts.
              </p>
              <ul>
                <li>6 chapters with latest research</li>
                <li>Multimodal perception systems</li>
                <li>Task planning and execution</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="summary-features">
          <h2>Key Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              <h4>Personalized Learning</h4>
              <p>Adaptive content delivery based on your background and learning pace preferences</p>
            </div>
            <div className="feature-card">
              <h4>AI-Powered Assistance</h4>
              <p>Intelligent chatbot for contextual answers to your questions</p>
            </div>
            <div className="feature-card">
              <h4>Multilingual Support</h4>
              <p>Content available in English and Urdu to support diverse learners</p>
            </div>
            <div className="feature-card">
              <h4>Progress Tracking</h4>
              <p>Comprehensive tracking of your learning journey across all modules</p>
            </div>
          </div>
        </div>
        
        <style jsx>{`
          .textbook-summary-container {
            padding: 2rem 1rem;
            max-width: 1200px;
            margin: 0 auto;
          }
          
          .summary-header {
            text-align: center;
            margin-bottom: 3rem;
          }
          
          .summary-header h1 {
            color: #2196F3;
            margin-bottom: 1rem;
          }
          
          .summary-overview {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            margin-bottom: 3rem;
          }
          
          .overall-progress {
            flex: 1;
            min-width: 300px;
            text-align: center;
          }
          
          .progress-circle {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 0 auto;
          }
          
          .progress-circle svg {
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
          }
          
          .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
          }
          
          .percentage {
            display: block;
            font-size: 2.5rem;
            font-weight: bold;
            color: #2196F3;
          }
          
          .count {
            font-size: 1rem;
            color: #666;
          }
          
          .progress-breakdown {
            flex: 2;
            min-width: 300px;
          }
          
          .module-progress {
            margin-bottom: 1.5rem;
          }
          
          .module-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
          }
          
          .module-header h3 {
            margin: 0;
            font-size: 1.2rem;
          }
          
          .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #eee;
            border-radius: 10px;
            overflow: hidden;
          }
          
          .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.5s ease;
          }
          
          .summary-content h2, .summary-features h2 {
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 0.75rem;
            margin-top: 2rem;
          }
          
          .modules-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
          }
          
          .module-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #eee;
          }
          
          .module-card h3 {
            color: #2196F3;
            margin-top: 0;
          }
          
          .module-card ul {
            padding-left: 1.5rem;
          }
          
          .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
          }
          
          .feature-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #eee;
            text-align: center;
          }
          
          .feature-card h4 {
            color: #2196F3;
            margin-top: 0;
          }
        `}</style>
      </div>
    </Layout>
  );
};

export default TextbookSummary;
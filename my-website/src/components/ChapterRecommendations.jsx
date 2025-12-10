// src/components/ChapterRecommendations.jsx
import React from 'react';
import { useUserProfile } from '@site/src/contexts/UserProfileContext';
import Link from '@docusaurus/Link';

// Define the recommended order of chapters across modules
const CHAPTER_SEQUENCE = [
  'chapter-1-intro-ros',
  'chapter-2-nodes-communication',
  'chapter-3-topics-publishers',
  'chapter-4-services-actions',
  'chapter-5-parameters-launch',
  'chapter-6-packages-build',
  'chapter-7-debugging-tools',
  'chapter-8-gazebo-simulation',
  'chapter-9-robot-modeling',
  'chapter-10-physics-engines',
  'chapter-11-unity-robotics',
  'chapter-12-advanced-scenarios',
  'chapter-13-sim-to-real',
  'chapter-14-nvidia-isaac',
  'chapter-15-perception-systems',
  'chapter-16-motion-planning',
  'chapter-17-robot-learning',
  'chapter-18-navigation-slam',
  'chapter-19-manipulation-grasping',
  'chapter-20-multi-robot',
  'chapter-21-ai-integration',
  'chapter-22-vision-language-action',
  'chapter-23-embodied-ai',
  'chapter-24-human-robot-interaction',
  'chapter-25-multimodal-perception',
  'chapter-26-task-planning',
  'chapter-27-advanced-intelligence'
];

// Chapter titles mapping
const CHAPTER_TITLES = {
  'chapter-1-intro-ros': 'Chapter 1 - Introduction to ROS 2 Architecture',
  'chapter-2-nodes-communication': 'Chapter 2 - Nodes and Communication Primitives',
  'chapter-3-topics-publishers': 'Chapter 3 - Topics and Publishers/Subscribers',
  'chapter-4-services-actions': 'Chapter 4 - Services and Actions',
  'chapter-5-parameters-launch': 'Chapter 5 - Parameters and Launch Systems',
  'chapter-6-packages-build': 'Chapter 6 - ROS 2 Packages and Build System',
  'chapter-7-debugging-tools': 'Chapter 7 - Debugging and Visualization Tools',
  'chapter-8-gazebo-simulation': 'Chapter 8 - Gazebo Simulation Environment',
  'chapter-9-robot-modeling': 'Chapter 9 - Robot Modeling and URDF',
  'chapter-10-physics-engines': 'Chapter 10 - Physics Engines and Sensor Simulation',
  'chapter-11-unity-robotics': 'Chapter 11 - Unity Integration for Robotics',
  'chapter-12-advanced-scenarios': 'Chapter 12 - Advanced Simulation Scenarios',
  'chapter-13-sim-to-real': 'Chapter 13 - Connecting Simulation to Reality',
  'chapter-14-nvidia-isaac': 'Chapter 14 - Introduction to NVIDIA Isaac Sim',
  'chapter-15-perception-systems': 'Chapter 15 - Perception Systems and Computer Vision',
  'chapter-16-motion-planning': 'Chapter 16 - Motion Planning and Pathfinding',
  'chapter-17-robot-learning': 'Chapter 17 - Robot Learning and Reinforcement Learning',
  'chapter-18-navigation-slam': 'Chapter 18 - Navigation and Mapping (SLAM)',
  'chapter-19-manipulation-grasping': 'Chapter 19 - Manipulation and Grasping',
  'chapter-20-multi-robot': 'Chapter 20 - Multi-Robot Systems',
  'chapter-21-ai-integration': 'Chapter 21 - AI Integration and Decision Making',
  'chapter-22-vision-language-action': 'Chapter 22 - Introduction to Vision-Language-Action Models',
  'chapter-23-embodied-ai': 'Chapter 23 - Embodied AI and Reasoning',
  'chapter-24-human-robot-interaction': 'Chapter 24 - Human-Robot Interaction and Communication',
  'chapter-25-multimodal-perception': 'Chapter 25 - Multimodal Perception',
  'chapter-26-task-planning': 'Chapter 26 - Task Planning and Execution',
  'chapter-27-advanced-intelligence': 'Chapter 27 - Advanced Embodied Intelligence'
};

// Map chapter IDs to module
const CHAPTER_MODULE_MAP = {
  'chapter-1-intro-ros': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-2-nodes-communication': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-3-topics-publishers': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-4-services-actions': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-5-parameters-launch': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-6-packages-build': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-7-debugging-tools': { module: 'module-1-ros', moduleTitle: 'Module 1: The Robotic Nervous System (ROS 2)' },
  'chapter-8-gazebo-simulation': { module: 'module-2-simulation', moduleTitle: 'Module 2: The Digital Twin (Gazebo & Unity)' },
  'chapter-9-robot-modeling': { module: 'module-2-simulation', moduleTitle: 'Module 2: The Digital Twin (Gazebo & Unity)' },
  'chapter-10-physics-engines': { module: 'module-2-simulation', moduleTitle: 'Module 2: The Digital Twin (Gazebo & Unity)' },
  'chapter-11-unity-robotics': { module: 'module-2-simulation', moduleTitle: 'Module 2: The Digital Twin (Gazebo & Unity)' },
  'chapter-12-advanced-scenarios': { module: 'module-2-simulation', moduleTitle: 'Module 2: The Digital Twin (Gazebo & Unity)' },
  'chapter-13-sim-to-real': { module: 'module-2-simulation', moduleTitle: 'Module 2: The Digital Twin (Gazebo & Unity)' },
  'chapter-14-nvidia-isaac': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-15-perception-systems': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-16-motion-planning': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-17-robot-learning': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-18-navigation-slam': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-19-manipulation-grasping': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-20-multi-robot': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-21-ai-integration': { module: 'module-3-ai', moduleTitle: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' },
  'chapter-22-vision-language-action': { module: 'module-4-vla', moduleTitle: 'Module 4: Vision-Language-Action (VLA)' },
  'chapter-23-embodied-ai': { module: 'module-4-vla', moduleTitle: 'Module 4: Vision-Language-Action (VLA)' },
  'chapter-24-human-robot-interaction': { module: 'module-4-vla', moduleTitle: 'Module 4: Vision-Language-Action (VLA)' },
  'chapter-25-multimodal-perception': { module: 'module-4-vla', moduleTitle: 'Module 4: Vision-Language-Action (VLA)' },
  'chapter-26-task-planning': { module: 'module-4-vla', moduleTitle: 'Module 4: Vision-Language-Action (VLA)' },
  'chapter-27-advanced-intelligence': { module: 'module-4-vla', moduleTitle: 'Module 4: Vision-Language-Action (VLA)' }
};

const ChapterRecommendations = () => {
  const { profile } = useUserProfile();
  const completedChapters = new Set(profile.completedChapters);
  
  // Find next recommended chapter based on sequence and progress
  const nextChapter = CHAPTER_SEQUENCE.find(chapterId => !completedChapters.has(chapterId));
  
  // Find recently completed chapters
  const recentlyCompleted = CHAPTER_SEQUENCE
    .filter(chapterId => completedChapters.has(chapterId))
    .slice(-3) // Get last 3 completed
    .reverse(); // Reverse to show most recent first
  
  // Find chapters that are not yet completed
  const remainingChapters = CHAPTER_SEQUENCE.filter(chapterId => !completedChapters.has(chapterId));
  
  // Get chapter in progress (not completed but not the next one either)
  const chaptersInProgress = CHAPTER_SEQUENCE
    .filter(chapterId => 
      completedChapters.has(chapterId) && 
      CHAPTER_SEQUENCE.indexOf(chapterId) < CHAPTER_SEQUENCE.indexOf(nextChapter)
    )
    .slice(-2); // Most recent 2 in progress

  return (
    <div className="chapter-recommendations">
      <div className="recommendations-container">
        {nextChapter && (
          <div className="recommendation-card next-chapter">
            <h3>Next Recommended Chapter</h3>
            <div className="chapter-info">
              <h4>{CHAPTER_TITLES[nextChapter]}</h4>
              <p>{CHAPTER_MODULE_MAP[nextChapter]?.moduleTitle}</p>
              <Link 
                className="btn btn-primary"
                to={`/docs/modules/${CHAPTER_MODULE_MAP[nextChapter].module}/${nextChapter}/index`}
              >
                Start Chapter
              </Link>
            </div>
          </div>
        )}
        
        <div className="recommendation-card recently-completed">
          <h3>Recently Completed</h3>
          <div className="chapter-list">
            {recentlyCompleted.length > 0 ? (
              recentlyCompleted.map(chapterId => (
                <div key={chapterId} className="chapter-item completed">
                  <h4>{CHAPTER_TITLES[chapterId]}</h4>
                  <p>{CHAPTER_MODULE_MAP[chapterId]?.moduleTitle}</p>
                </div>
              ))
            ) : (
              <p className="no-data">No chapters completed yet. Start with the first chapter!</p>
            )}
          </div>
        </div>
        
        {remainingChapters.length > 0 && (
          <div className="recommendation-card remaining-chapters">
            <h3>Continue Learning</h3>
            <p>Chapters that might interest you based on your progress:</p>
            <div className="chapter-list">
              {remainingChapters.slice(0, 5).map(chapterId => (
                <div key={chapterId} className="chapter-item">
                  <h4>{CHAPTER_TITLES[chapterId]}</h4>
                  <p>{CHAPTER_MODULE_MAP[chapterId]?.moduleTitle}</p>
                  <Link 
                    to={`/docs/modules/${CHAPTER_MODULE_MAP[chapterId].module}/${chapterId}/index`}
                  >
                    {completedChapters.has(chapterId) ? 'Review' : 'View'}
                  </Link>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {chaptersInProgress.length > 0 && (
          <div className="recommendation-card in-progress">
            <h3>Chapters in Progress</h3>
            <div className="chapter-list">
              {chaptersInProgress.map(chapterId => (
                <div key={chapterId} className="chapter-item in-progress">
                  <h4>{CHAPTER_TITLES[chapterId]}</h4>
                  <p>{CHAPTER_MODULE_MAP[chapterId]?.moduleTitle}</p>
                  <Link 
                    to={`/docs/modules/${CHAPTER_MODULE_MAP[chapterId].module}/${chapterId}/index`}
                  >
                    Continue
                  </Link>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <style jsx>{`
        .chapter-recommendations {
          padding: 2rem 0;
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 1rem;
        }
        
        .recommendations-container {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
        }
        
        .recommendation-card {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
          box-shadow: 0 2px 10px rgba(0,0,0,0.05);
          border: 1px solid #eee;
        }
        
        .next-chapter {
          border-left: 4px solid #4CAF50;
        }
        
        .in-progress {
          border-left: 4px solid #FF9800;
        }
        
        .recommendation-card h3 {
          margin-top: 0;
          color: #2196F3;
          border-bottom: 1px solid #eee;
          padding-bottom: 0.75rem;
        }
        
        .chapter-info {
          padding: 1rem 0;
        }
        
        .chapter-info h4 {
          margin: 0.5rem 0;
          color: #333;
        }
        
        .chapter-info p {
          color: #666;
          margin: 0.25rem 0;
        }
        
        .chapter-list {
          margin-top: 1rem;
        }
        
        .chapter-item {
          padding: 1rem 0;
          border-bottom: 1px solid #f0f0f0;
        }
        
        .chapter-item:last-child {
          border-bottom: none;
        }
        
        .chapter-item h4 {
          margin: 0.5rem 0;
          color: #333;
        }
        
        .chapter-item p {
          color: #666;
          margin: 0.25rem 0;
          font-size: 0.9rem;
        }
        
        .chapter-item a {
          color: #2196F3;
          text-decoration: none;
          font-weight: 500;
          font-size: 0.9rem;
        }
        
        .chapter-item a:hover {
          text-decoration: underline;
        }
        
        .completed h4 {
          color: #4CAF50;
        }
        
        .in-progress h4 {
          color: #FF9800;
        }
        
        .no-data {
          color: #666;
          font-style: italic;
        }
        
        .btn {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s ease;
          text-decoration: none;
          display: inline-block;
          text-align: center;
        }
        
        .btn-primary {
          background-color: #2196F3;
          color: white;
        }
        
        .btn-primary:hover {
          background-color: #1976D2;
        }
      `}</style>
    </div>
  );
};

export default ChapterRecommendations;
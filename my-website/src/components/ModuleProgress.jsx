// src/components/ModuleProgress.jsx
import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

// Define the chapters in each module for progress calculation
const MODULE_CHAPTERS = {
  'module-1-ros': [
    'chapter-1-intro-ros',
    'chapter-2-nodes-communication',
    'chapter-3-topics-publishers',
    'chapter-4-services-actions',
    'chapter-5-parameters-launch',
    'chapter-6-packages-build',
    'chapter-7-debugging-tools'
  ],
  'module-2-simulation': [
    'chapter-8-gazebo-simulation',
    'chapter-9-robot-modeling',
    'chapter-10-physics-engines',
    'chapter-11-unity-robotics',
    'chapter-12-advanced-scenarios',
    'chapter-13-sim-to-real'
  ],
  'module-3-ai': [
    'chapter-14-nvidia-isaac',
    'chapter-15-perception-systems',
    'chapter-16-motion-planning',
    'chapter-17-robot-learning',
    'chapter-18-navigation-slam',
    'chapter-19-manipulation-grasping',
    'chapter-20-multi-robot',
    'chapter-21-ai-integration'
  ],
  'module-4-vla': [
    'chapter-22-vision-language-action',
    'chapter-23-embodied-ai',
    'chapter-24-human-robot-interaction',
    'chapter-25-multimodal-perception',
    'chapter-26-task-planning',
    'chapter-27-advanced-intelligence'
  ]
};

// Inner component that only runs in browser
const ModuleProgressInner = ({ moduleId }) => {
  const moduleChapters = MODULE_CHAPTERS[moduleId] || [];
  const [progress, setProgress] = useState(0);
  const [completedCount, setCompletedCount] = useState(0);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.RoboticsTextbookProgress) {
      const getProgressPercentage = window.RoboticsTextbookProgress.getProgressPercentage;
      const completedChapters = window.RoboticsTextbookProgress.completedChapters || [];

      const progressValue = getProgressPercentage(moduleChapters);
      const completed = completedChapters.filter(id => moduleChapters.includes(id)).length;

      setProgress(progressValue);
      setCompletedCount(completed);
    }
  }, [moduleId, moduleChapters]);

  return (
    <div className="module-progress">
      <div className="progress-bar-container">
        <div className="progress-text">Module Progress: {progress}%</div>
        <div className="progress-bar">
          <div
            className="progress-bar-fill"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="progress-stats">
          {completedCount} of {moduleChapters.length} chapters completed
        </div>
      </div>

      <style jsx>{`
        .module-progress {
          margin: 1rem 0;
          padding: 1rem;
          border: 1px solid #444;
          border-radius: 4px;
          background-color: #2d2d2d;
          color: white;
        }

        .progress-bar-container {
          width: 100%;
        }

        .progress-text {
          margin-bottom: 0.5rem;
          font-weight: bold;
          color: white;
        }

        .progress-bar {
          width: 100%;
          height: 20px;
          background-color: #444;
          border-radius: 10px;
          overflow: hidden;
          margin-bottom: 0.5rem;
        }

        .progress-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, #4CAF50, #8BC34A);
          transition: width 0.3s ease;
        }

        .progress-stats {
          font-size: 0.9rem;
          color: #ccc;
          text-align: center;
        }
      `}</style>
    </div>
  );
};

const ModuleProgress = ({ moduleId }) => {
  return (
    <BrowserOnly>
      {() => <ModuleProgressInner moduleId={moduleId} />}
    </BrowserOnly>
  );
};

export default ModuleProgress;
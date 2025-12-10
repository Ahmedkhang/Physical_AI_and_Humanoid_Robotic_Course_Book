// src/components/ChapterProgress.jsx
import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

// Function to get progress API from window
const getProgressAPI = () => {
  if (typeof window !== 'undefined' && window.RoboticsTextbookProgress) {
    return window.RoboticsTextbookProgress;
  }
  return null;
};

// Inner component that only runs in browser
const ChapterProgressInner = ({ chapterId }) => {
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const progressAPI = getProgressAPI();
    if (progressAPI) {
      setIsComplete(progressAPI.isChapterComplete(chapterId));
    }
  }, [chapterId]);

  const toggleCompletion = () => {
    const progressAPI = getProgressAPI();
    if (progressAPI) {
      if (isComplete) {
        progressAPI.markChapterIncomplete(chapterId);
        setIsComplete(false);
      } else {
        progressAPI.markChapterComplete(chapterId);
        setIsComplete(true);
      }
    }
  };

  if (!getProgressAPI()) {
    // If API is not available, show placeholder
    return (
      <div className="chapter-progress">
        <div className="progress-button-placeholder">Loading...</div>
        <style jsx>{`
          .chapter-progress {
            margin: 1rem 0;
            text-align: right;
          }
          .progress-button-placeholder {
            padding: 8px 16px;
            border: 2px solid #ccc;
            border-radius: 4px;
            background-color: #f5f5f5;
            color: #999;
            font-size: 14px;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="chapter-progress">
      <button
        onClick={toggleCompletion}
        className={`progress-button ${isComplete ? 'completed' : 'incomplete'}`}
        aria-label={isComplete ? 'Mark as incomplete' : 'Mark as complete'}
      >
        {isComplete ? '✓ Completed' : '○ Mark Complete'}
      </button>
      <style jsx>{`
        .chapter-progress {
          margin: 1rem 0;
          text-align: right;
        }

        .progress-button {
          padding: 8px 16px;
          border: 2px solid #2196F3;
          border-radius: 4px;
          background-color: white;
          color: #2196F3;
          cursor: pointer;
          font-size: 14px;
          transition: all 0.2s ease;
        }

        .progress-button:hover {
          background-color: #2196F3;
          color: white;
        }

        .progress-button.completed {
          background-color: #4CAF50;
          border-color: #4CAF50;
          color: white;
        }

        .progress-button.completed:hover {
          background-color: #388E3C;
          border-color: #388E3C;
        }
      `}</style>
    </div>
  );
};

const ChapterProgress = ({ chapterId }) => {
  return (
    <BrowserOnly>
      {() => <ChapterProgressInner chapterId={chapterId} />}
    </BrowserOnly>
  );
};

export default ChapterProgress;
// src/plugins/progress-tracker.js
import React, { useState, useEffect } from 'react';

const ProgressTracker = () => {
  const [completedChapters, setCompletedChapters] = useState([]);

  useEffect(() => {
    // Load completed chapters from localStorage
    const savedProgress = localStorage.getItem('robotics-textbook-progress');
    if (savedProgress) {
      setCompletedChapters(JSON.parse(savedProgress));
    }
  }, []);

  useEffect(() => {
    // Save completed chapters to localStorage
    localStorage.setItem('robotics-textbook-progress', JSON.stringify(completedChapters));
  }, [completedChapters]);

  const markChapterComplete = (chapterId) => {
    if (!completedChapters.includes(chapterId)) {
      setCompletedChapters([...completedChapters, chapterId]);
    }
  };

  const markChapterIncomplete = (chapterId) => {
    setCompletedChapters(completedChapters.filter(id => id !== chapterId));
  };

  const isChapterComplete = (chapterId) => {
    return completedChapters.includes(chapterId);
  };

  const getProgressPercentage = (moduleChapters) => {
    const completedInModule = moduleChapters.filter(chapterId => 
      completedChapters.includes(chapterId)
    ).length;
    return Math.round((completedInModule / moduleChapters.length) * 100);
  };

  // Provide the functions and state to other components
  window.RoboticsTextbookProgress = {
    markChapterComplete,
    markChapterIncomplete,
    isChapterComplete,
    getProgressPercentage,
    completedChapters
  };

  return null; // This component doesn't render anything visible
};

export default ProgressTracker;
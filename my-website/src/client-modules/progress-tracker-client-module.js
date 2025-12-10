// src/client-modules/progress-tracker-client-module.js
// Client module to initialize the progress tracking functionality

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

// Add a global event listener to mark chapters as complete when visited
window.addEventListener('load', () => {
  // Identify the current chapter from the URL
  const pathParts = window.location.pathname.split('/');
  const chapterPartIndex = pathParts.findIndex(part => part.includes('chapter-'));
  if (chapterPartIndex !== -1) {
    const chapterId = pathParts[chapterPartIndex];
    
    // Mark the chapter as visited (but not necessarily completed)
    // The user still needs to explicitly mark it as complete
    window.currentChapterId = chapterId;
  }
});
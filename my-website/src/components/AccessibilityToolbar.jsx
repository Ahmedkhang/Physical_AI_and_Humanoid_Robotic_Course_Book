// src/components/AccessibilityToolbar.jsx
import React, { useState, useEffect } from 'react';

const AccessibilityToolbar = () => {
  const [fontSize, setFontSize] = useState('medium');
  const [contrastMode, setContrastMode] = useState('normal');
  const [isReadingMode, setIsReadingMode] = useState(false);
  const [isAudioMode, setIsAudioMode] = useState(false);
  
  // Apply styles based on current settings
  useEffect(() => {
    document.body.className = document.body.className.replace(/font-size-\w+/, '');
    document.body.classList.add(`font-size-${fontSize}`);
    
    // Apply contrast mode if needed
    if (contrastMode === 'high') {
      document.body.classList.add('high-contrast');
    } else {
      document.body.classList.remove('high-contrast');
    }
    
    // Apply reading mode if needed
    if (isReadingMode) {
      document.body.classList.add('reading-mode');
    } else {
      document.body.classList.remove('reading-mode');
    }
  }, [fontSize, contrastMode, isReadingMode]);
  
  const increaseFontSize = () => {
    const sizes = ['small', 'medium', 'large', 'xlarge'];
    const currentIndex = sizes.indexOf(fontSize);
    if (currentIndex < sizes.length - 1) {
      setFontSize(sizes[currentIndex + 1]);
    }
  };
  
  const decreaseFontSize = () => {
    const sizes = ['small', 'medium', 'large', 'xlarge'];
    const currentIndex = sizes.indexOf(fontSize);
    if (currentIndex > 0) {
      setFontSize(sizes[currentIndex - 1]);
    }
  };
  
  const toggleContrast = () => {
    setContrastMode(contrastMode === 'normal' ? 'high' : 'normal');
  };
  
  const toggleReadingMode = () => {
    setIsReadingMode(!isReadingMode);
  };
  
  const toggleAudioMode = () => {
    setIsAudioMode(!isAudioMode);
    // In a real implementation, this would trigger text-to-speech
  };
  
  return (
    <div className="accessibility-toolbar" role="region" aria-label="Accessibility tools">
      <div className="toolbar-container">
        <h3>Accessibility Tools</h3>
        <div className="toolbar-options">
          <div className="toolbar-group">
            <button 
              onClick={decreaseFontSize} 
              aria-label="Decrease font size"
              title="Decrease font size"
            >
              A-
            </button>
            <span className="font-size-indicator">{fontSize}</span>
            <button 
              onClick={increaseFontSize} 
              aria-label="Increase font size"
              title="Increase font size"
            >
              A+
            </button>
          </div>
          
          <div className="toolbar-group">
            <button 
              onClick={toggleContrast}
              className={contrastMode === 'high' ? 'active' : ''}
              aria-pressed={contrastMode === 'high'}
              title={contrastMode === 'high' ? "Disable high contrast" : "Enable high contrast"}
            >
              {contrastMode === 'high' ? 'Normal Contrast' : 'High Contrast'}
            </button>
          </div>
          
          <div className="toolbar-group">
            <button 
              onClick={toggleReadingMode}
              className={isReadingMode ? 'active' : ''}
              aria-pressed={isReadingMode}
              title={isReadingMode ? "Exit reading mode" : "Enter reading mode"}
            >
              {isReadingMode ? 'Exit Reading Mode' : 'Reading Mode'}
            </button>
          </div>
          
          <div className="toolbar-group">
            <button 
              onClick={toggleAudioMode}
              className={isAudioMode ? 'active' : ''}
              aria-pressed={isAudioMode}
              title={isAudioMode ? "Stop audio" : "Start audio"}
            >
              {isAudioMode ? '🔊 Audio On' : '🔇 Audio Off'}
            </button>
          </div>
          
          <div className="toolbar-group">
            <a href="#main-content" className="skip-link">
              Skip to main content
            </a>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .accessibility-toolbar {
          position: sticky;
          top: 0;
          background-color: #2d2d2d;
          padding: 0.5rem;
          border-bottom: 2px solid #2196F3;
          z-index: 1000;
          color: white;
        }

        .toolbar-container {
          max-width: 1200px;
          margin: 0 auto;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .toolbar-container h3 {
          margin: 0;
          font-size: 1rem;
          color: white;
        }

        .toolbar-options {
          display: flex;
          gap: 1rem;
          align-items: center;
        }

        .toolbar-group {
          display: flex;
          gap: 0.5rem;
          align-items: center;
        }

        button {
          padding: 0.5rem;
          border: 1px solid #555;
          background-color: #444;
          color: white;
          cursor: pointer;
          border-radius: 4px;
          font-size: 0.8rem;
        }

        button:hover {
          background-color: #555;
        }

        button.active {
          background-color: #2196F3;
          color: white;
          border-color: #2196F3;
        }

        .font-size-indicator {
          padding: 0 0.5rem;
          font-weight: bold;
          color: white;
        }

        .skip-link {
          padding: 0.25rem 0.5rem;
          background-color: #2196F3;
          color: white;
          text-decoration: none;
          border-radius: 4px;
          font-size: 0.8rem;
        }

        .skip-link:hover {
          background-color: #1976D2;
        }
        
        /* Accessibility styles */
        .high-contrast {
          filter: contrast(1.5) brightness(1.1);
        }
        
        .reading-mode {
          max-width: 800px;
          margin: 0 auto;
          padding: 0 1rem;
        }
        
        .font-size-small {
          font-size: 0.9rem;
        }
        
        .font-size-medium {
          font-size: 1rem;
        }
        
        .font-size-large {
          font-size: 1.2rem;
        }
        
        .font-size-xlarge {
          font-size: 1.5rem;
        }
      `}</style>
    </div>
  );
};

export default AccessibilityToolbar;
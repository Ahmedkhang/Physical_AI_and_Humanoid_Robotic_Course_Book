// src/components/PersonalizationDashboard.jsx
import React from 'react';
import { useUserProfile } from '@site/src/contexts/UserProfileContext';

const PersonalizationDashboard = () => {
  const { profile, isSurveyCompleted, updatePreference, markChapterComplete, markChapterIncomplete } = useUserProfile();
  
  const learningPaceOptions = [
    { value: 'slow', label: 'Slow' },
    { value: 'moderate', label: 'Moderate' },
    { value: 'fast', label: 'Fast' }
  ];
  
  const fontSizeOptions = [
    { value: 'small', label: 'Small' },
    { value: 'medium', label: 'Medium' },
    { value: 'large', label: 'Large' }
  ];
  
  return (
    <div className="personalization-dashboard">
      <div className="dashboard-container">
        <h1>Personalization Dashboard</h1>
        
        <div className="dashboard-grid">
          {/* Learning Pace Settings */}
          <div className="dashboard-card">
            <h2>Learning Pace</h2>
            <div className="setting-item">
              <label htmlFor="learningPace">Your preferred learning pace:</label>
              <select
                id="learningPace"
                value={profile.learningPace}
                onChange={(e) => updatePreference('learningPace', e.target.value)}
              >
                {learningPaceOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <p className="setting-description">
              Adjust this setting to control the depth and speed of content delivery.
            </p>
          </div>
          
          {/* Display Preferences */}
          <div className="dashboard-card">
            <h2>Display Preferences</h2>
            <div className="setting-item">
              <label htmlFor="fontSize">Font size:</label>
              <select
                id="fontSize"
                value={profile.preferences.fontSize}
                onChange={(e) => updatePreference('preferences', {
                  ...profile.preferences,
                  fontSize: e.target.value
                })}
              >
                {fontSizeOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="setting-item checkbox-item">
              <label htmlFor="darkMode">
                <input
                  type="checkbox"
                  id="darkMode"
                  checked={profile.preferences.darkMode}
                  onChange={(e) => updatePreference('preferences', {
                    ...profile.preferences,
                    darkMode: e.target.checked
                  })}
                />
                Enable dark mode
              </label>
            </div>
            <div className="setting-item checkbox-item">
              <label htmlFor="notifications">
                <input
                  type="checkbox"
                  id="notifications"
                  checked={profile.preferences.notifications}
                  onChange={(e) => updatePreference('preferences', {
                    ...profile.preferences,
                    notifications: e.target.checked
                  })}
                />
                Enable learning reminders
              </label>
            </div>
          </div>
          
          {/* Learning Progress */}
          <div className="dashboard-card">
            <h2>Learning Progress</h2>
            <div className="progress-summary">
              <div className="progress-stat">
                <span className="stat-number">{profile.completedChapters.length}</span>
                <span className="stat-label">Chapters Completed</span>
              </div>
              <div className="progress-stat">
                <span className="stat-number">
                  {profile.completedChapters.length > 0 
                    ? Math.round((profile.completedChapters.length / 27) * 100) 
                    : 0}%
                </span>
                <span className="stat-label">Overall Progress</span>
              </div>
            </div>
            <div className="recent-activity">
              <h3>Recent Activity</h3>
              <p>Keep up the great work! Your learning pace is well-suited to your goals.</p>
            </div>
          </div>
          
          {/* Survey Status */}
          <div className="dashboard-card">
            <h2>Background Survey</h2>
            {isSurveyCompleted ? (
              <div className="survey-status completed">
                <div className="status-icon">✓</div>
                <div>
                  <h3>Survey Completed</h3>
                  <p>Thank you for completing the background survey. Your learning experience is personalized based on your responses.</p>
                </div>
              </div>
            ) : (
              <div className="survey-status incomplete">
                <div className="status-icon">!</div>
                <div>
                  <h3>Survey Incomplete</h3>
                  <p>Complete the background survey to personalize your learning experience and receive recommendations tailored to your needs.</p>
                  <button 
                    className="btn btn-primary"
                    onClick={() => window.location.href = '/survey'}
                  >
                    Complete Survey
                  </button>
                </div>
              </div>
            )}
          </div>
          
          {/* Language Preferences */}
          <div className="dashboard-card">
            <h2>Language Settings</h2>
            <div className="setting-item">
              <label htmlFor="language">Interface language:</label>
              <select
                id="language"
                value={profile.languagePreference}
                onChange={(e) => updatePreference('languagePreference', e.target.value)}
              >
                <option value="en">English</option>
                <option value="ur">Urdu</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .personalization-dashboard {
          padding: 2rem 0;
        }
        
        .dashboard-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 1rem;
        }
        
        h1 {
          text-align: center;
          margin-bottom: 2rem;
          color: #333;
        }
        
        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
          gap: 1.5rem;
        }
        
        .dashboard-card {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
          box-shadow: 0 2px 10px rgba(0,0,0,0.05);
          border: 1px solid #eee;
        }
        
        .dashboard-card h2 {
          margin-top: 0;
          color: #2196F3;
          border-bottom: 1px solid #eee;
          padding-bottom: 0.75rem;
        }
        
        .setting-item {
          margin-bottom: 1rem;
        }
        
        .setting-item label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
          color: #333;
        }
        
        .setting-item select {
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 1rem;
        }
        
        .checkbox-item {
          display: flex;
          align-items: center;
        }
        
        .checkbox-item input {
          margin-right: 0.5rem;
        }
        
        .setting-description {
          font-size: 0.9rem;
          color: #666;
          margin-top: 0.25rem;
        }
        
        .progress-summary {
          display: flex;
          justify-content: space-around;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid #eee;
        }
        
        .progress-stat {
          text-align: center;
        }
        
        .stat-number {
          display: block;
          font-size: 2rem;
          font-weight: bold;
          color: #2196F3;
        }
        
        .stat-label {
          font-size: 0.9rem;
          color: #666;
        }
        
        .recent-activity {
          margin-top: 1rem;
        }
        
        .recent-activity h3 {
          margin-top: 0;
          margin-bottom: 0.5rem;
          color: #555;
        }
        
        .survey-status {
          display: flex;
          align-items: flex-start;
        }
        
        .status-icon {
          font-size: 2rem;
          margin-right: 1rem;
          min-width: 40px;
          text-align: center;
        }
        
        .survey-status.completed .status-icon {
          color: #4CAF50;
        }
        
        .survey-status.incomplete .status-icon {
          color: #FF9800;
        }
        
        .survey-status div:last-child {
          flex: 1;
        }
        
        .btn {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s ease;
          text-decoration: none;
          display: inline-block;
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

export default PersonalizationDashboard;
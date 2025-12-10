// src/components/LanguageToggle.jsx
import React from 'react';
import { useLanguage } from '@site/src/contexts/LanguageContext';

const LanguageToggle = () => {
  const { currentLanguage, availableLanguages, changeLanguage } = useLanguage();
  
  // Define language names
  const languageNames = {
    en: 'English',
    ur: 'Urdu'
  };
  
  // Handle language change
  const handleLanguageChange = (e) => {
    changeLanguage(e.target.value);
  };
  
  return (
    <div className="language-toggle">
      <label htmlFor="language-select">Language:</label>
      <select
        id="language-select"
        value={currentLanguage}
        onChange={handleLanguageChange}
      >
        {availableLanguages.map(langCode => (
          <option key={langCode} value={langCode}>
            {languageNames[langCode]}
          </option>
        ))}
      </select>
      
      <style jsx>{`
        .language-toggle {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        label {
          font-size: 0.9rem;
          color: white;
        }

        select {
          padding: 0.25rem 0.5rem;
          background-color: #2d2d2d;
          color: white;
          border: 1px solid #555;
          border-radius: 4px;
          font-size: 0.9rem;
        }

        select option {
          background-color: #2d2d2d;
          color: white;
        }
      `}</style>
    </div>
  );
};

export default LanguageToggle;
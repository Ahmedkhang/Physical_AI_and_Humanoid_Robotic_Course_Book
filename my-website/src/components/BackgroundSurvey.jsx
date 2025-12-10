// src/components/BackgroundSurvey.jsx
import React, { useState } from 'react';
import { useUserProfile } from '@site/src/contexts/UserProfileContext';

const BackgroundSurvey = () => {
  const { setSurveyCompleted, updatePreference } = useUserProfile();
  const [currentStep, setCurrentStep] = useState(0);
  const [surveyData, setSurveyData] = useState({
    // Demographics
    experienceLevel: '',
    fieldOfStudy: '',
    primaryGoal: '',
    // Learning preferences
    learningPace: 'moderate',
    studyTime: '',
    // Background
    programmingExperience: '',
    roboticsExperience: '',
    mathBackground: '',
  });
  
  const surveySteps = [
    {
      title: "Tell us about yourself",
      fields: [
        {
          name: "experienceLevel",
          label: "What's your experience level with robotics?",
          type: "radio",
          options: [
            { value: "beginner", label: "Beginner - Just starting out" },
            { value: "intermediate", label: "Intermediate - Some experience" },
            { value: "advanced", label: "Advanced - Significant experience" },
            { value: "expert", label: "Expert - Professional/research level" }
          ]
        },
        {
          name: "fieldOfStudy",
          label: "What's your field of study or profession?",
          type: "select",
          options: [
            { value: "", label: "Select an option" },
            { value: "student", label: "Student" },
            { value: "engineer", label: "Robotics Engineer" },
            { value: "researcher", label: "Researcher" },
            { value: "developer", label: "Software Developer" },
            { value: "hobbyist", label: "Hobbyist/Maker" },
            { value: "other", label: "Other" }
          ]
        }
      ]
    },
    {
      title: "Learning Preferences",
      fields: [
        {
          name: "learningPace",
          label: "How would you describe your preferred learning pace?",
          type: "radio",
          options: [
            { value: "slow", label: "Slow - I prefer detailed explanations and take my time" },
            { value: "moderate", label: "Moderate - Good pace, balanced detail" },
            { value: "fast", label: "Fast - I prefer to move quickly through material" }
          ]
        },
        {
          name: "studyTime",
          label: "How much time can you dedicate to learning each week?",
          type: "radio",
          options: [
            { value: "less-2", label: "Less than 2 hours" },
            { value: "2-5", label: "2-5 hours" },
            { value: "5-10", label: "5-10 hours" },
            { value: "10-plus", label: "More than 10 hours" }
          ]
        }
      ]
    },
    {
      title: "Technical Background",
      fields: [
        {
          name: "programmingExperience",
          label: "What's your programming experience?",
          type: "radio",
          options: [
            { value: "none", label: "None or very basic" },
            { value: "basic", label: "Basic - Can write simple programs" },
            { value: "intermediate", label: "Intermediate - Comfortable with programming concepts" },
            { value: "advanced", label: "Advanced - Experienced with multiple languages" }
          ]
        },
        {
          name: "roboticsExperience",
          label: "What's your robotics experience?",
          type: "radio",
          options: [
            { value: "none", label: "None - First time working with robotics" },
            { value: "simulation", label: "Simulation - Only worked in simulation" },
            { value: "basic", label: "Basic hardware - Simple robots" },
            { value: "advanced", label: "Advanced - Complex robot systems" }
          ]
        },
        {
          name: "mathBackground",
          label: "What's your math background?",
          type: "radio",
          options: [
            { value: "algebra", label: "Algebra/Trigonometry level" },
            { value: "calculus", label: "Calculus level" },
            { value: "linear-algebra", label: "Linear algebra level" },
            { value: "advanced", label: "Advanced - Graduate level math" }
          ]
        }
      ]
    },
    {
      title: "Final Goal",
      fields: [
        {
          name: "primaryGoal",
          label: "What's your primary goal in learning about robotics?",
          type: "checkbox-group",
          options: [
            { value: "career", label: "Career advancement in robotics" },
            { value: "education", label: "Academic education or research" },
            { value: "project", label: "Personal or hobby projects" },
            { value: "research", label: "Research in AI/Robotics" },
            { value: "startup", label: "Starting a robotics company" }
          ]
        }
      ]
    }
  ];
  
  const handleInputChange = (name, value) => {
    setSurveyData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleNext = () => {
    if (currentStep < surveySteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Complete survey
      completeSurvey();
    }
  };
  
  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };
  
  const completeSurvey = () => {
    // Update the user profile with the survey data
    Object.keys(surveyData).forEach(key => {
      if (key === 'learningPace') {
        updatePreference('learningPace', surveyData[key]);
      } else {
        // We can save the survey data for later use
        // For now, we'll just store the learning pace preference
      }
    });
    
    setSurveyCompleted(true);
    alert("Thank you for completing the survey! Your learning experience will be personalized based on your responses.");
  };
  
  const currentStepData = surveySteps[currentStep];
  
  return (
    <div className="background-survey">
      <div className="survey-container">
        <div className="survey-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${((currentStep + 1) / surveySteps.length) * 100}%` }}
            ></div>
          </div>
          <div className="progress-text">
            Question {currentStep + 1} of {surveySteps.length}
          </div>
        </div>
        
        <h2>{currentStepData.title}</h2>
        
        <div className="survey-fields">
          {currentStepData.fields.map((field) => (
            <div key={field.name} className="survey-field">
              <label>{field.label}</label>
              
              {field.type === 'radio' && (
                <div className="radio-group">
                  {field.options.map((option) => (
                    <div key={option.value} className="radio-option">
                      <input
                        type="radio"
                        id={`${field.name}-${option.value}`}
                        name={field.name}
                        value={option.value}
                        checked={surveyData[field.name] === option.value}
                        onChange={(e) => handleInputChange(field.name, e.target.value)}
                      />
                      <label htmlFor={`${field.name}-${option.value}`}>
                        {option.label}
                      </label>
                    </div>
                  ))}
                </div>
              )}
              
              {field.type === 'select' && (
                <select
                  value={surveyData[field.name]}
                  onChange={(e) => handleInputChange(field.name, e.target.value)}
                >
                  {field.options.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              )}
              
              {field.type === 'checkbox-group' && (
                <div className="checkbox-group">
                  {field.options.map((option) => (
                    <div key={option.value} className="checkbox-option">
                      <input
                        type="checkbox"
                        id={`${field.name}-${option.value}`}
                        name={field.name}
                        value={option.value}
                        checked={surveyData[field.name]?.includes(option.value)}
                        onChange={(e) => {
                          const currentValue = surveyData[field.name] || [];
                          let newValue;
                          
                          if (e.target.checked) {
                            newValue = [...currentValue, option.value];
                          } else {
                            newValue = currentValue.filter(v => v !== option.value);
                          }
                          
                          handleInputChange(field.name, newValue);
                        }}
                      />
                      <label htmlFor={`${field.name}-${option.value}`}>
                        {option.label}
                      </label>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div className="survey-actions">
          <button 
            onClick={handlePrevious} 
            disabled={currentStep === 0}
            className="btn btn-secondary"
          >
            Previous
          </button>
          
          <button 
            onClick={handleNext}
            className="btn btn-primary"
          >
            {currentStep === surveySteps.length - 1 ? 'Complete Survey' : 'Next'}
          </button>
        </div>
      </div>
      
      <style jsx>{`
        .background-survey {
          padding: 2rem;
          max-width: 800px;
          margin: 0 auto;
        }
        
        .survey-container {
          background: white;
          border-radius: 8px;
          padding: 2rem;
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .survey-progress {
          margin-bottom: 2rem;
        }
        
        .progress-bar {
          width: 100%;
          height: 10px;
          background-color: #e0e0e0;
          border-radius: 5px;
          overflow: hidden;
          margin-bottom: 0.5rem;
        }
        
        .progress-fill {
          height: 100%;
          background: #2196F3;
          transition: width 0.3s ease;
        }
        
        .progress-text {
          text-align: center;
          font-size: 0.9rem;
          color: #666;
        }
        
        h2 {
          margin-top: 0;
          color: #333;
          border-bottom: 1px solid #eee;
          padding-bottom: 1rem;
        }
        
        .survey-fields {
          margin: 2rem 0;
        }
        
        .survey-field {
          margin-bottom: 1.5rem;
        }
        
        .survey-field label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 600;
          color: #333;
        }
        
        .radio-group, .checkbox-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .radio-option, .checkbox-option {
          display: flex;
          align-items: center;
          padding: 0.75rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .radio-option:hover, .checkbox-option:hover {
          border-color: #2196F3;
          background-color: #f5f9ff;
        }
        
        .radio-option input, .checkbox-option input {
          margin-right: 0.75rem;
        }
        
        select {
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 1rem;
        }
        
        .survey-actions {
          display: flex;
          justify-content: space-between;
          margin-top: 2rem;
        }
        
        .btn {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .btn-primary {
          background-color: #2196F3;
          color: white;
        }
        
        .btn-primary:hover {
          background-color: #1976D2;
        }
        
        .btn-secondary {
          background-color: #e0e0e0;
          color: #333;
        }
        
        .btn-secondary:hover {
          background-color: #bdbdbd;
        }
        
        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default BackgroundSurvey;
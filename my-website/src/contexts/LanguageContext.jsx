// src/contexts/LanguageContext.jsx
import React, { createContext, useContext, useReducer } from 'react';

// Define the initial state for language
const initialState = {
  currentLanguage: 'en',
  availableLanguages: ['en', 'ur'],
  loading: true,
};

// Define actions for the reducer
const actionTypes = {
  SET_LANGUAGE: 'SET_LANGUAGE',
  SET_LOADING: 'SET_LOADING',
};

// Reducer function
function languageReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_LANGUAGE:
      return {
        ...state,
        currentLanguage: action.payload,
      };
      
    case actionTypes.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
      };
      
    default:
      return state;
  }
}

// Create the context
const LanguageContext = createContext();

// Provider component
export function LanguageProvider({ children }) {
  const [state, dispatch] = useReducer(languageReducer, initialState);
  
  // Load language from localStorage on initialization
  React.useEffect(() => {
    const savedLanguage = localStorage.getItem('robotics-textbook-language');
    
    if (savedLanguage && initialState.availableLanguages.includes(savedLanguage)) {
      dispatch({ type: actionTypes.SET_LANGUAGE, payload: savedLanguage });
    } else {
      // Default to English if no preference is saved
      dispatch({ type: actionTypes.SET_LANGUAGE, payload: 'en' });
    }
    
    dispatch({ type: actionTypes.SET_LOADING, payload: false });
  }, []);
  
  // Save language to localStorage whenever it changes
  React.useEffect(() => {
    if (!state.loading) {
      localStorage.setItem('robotics-textbook-language', state.currentLanguage);
      
      // Update the HTML lang attribute
      document.documentElement.lang = state.currentLanguage;
    }
  }, [state.currentLanguage, state.loading]);
  
  // Function to change language
  const changeLanguage = (languageCode) => {
    if (state.availableLanguages.includes(languageCode)) {
      dispatch({ type: actionTypes.SET_LANGUAGE, payload: languageCode });
    } else {
      console.warn(`Language ${languageCode} is not available`);
    }
  };
  
  const value = {
    ...state,
    changeLanguage,
  };
  
  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

// Custom hook to use the language context
export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
// src/contexts/UserProfileContext.jsx
import React, { createContext, useContext, useReducer } from 'react';

// Define the initial state for user profile
const initialState = {
  profile: {
    id: null,
    languagePreference: 'en',
    learningPace: 'moderate', // 'slow', 'moderate', 'fast'
    completedChapters: [],
    preferences: {
      darkMode: false,
      fontSize: 'medium', // 'small', 'medium', 'large'
      notifications: true,
    },
  },
  isSurveyCompleted: false,
  loading: true,
};

// Define actions for the reducer
const actionTypes = {
  SET_USER_PROFILE: 'SET_USER_PROFILE',
  UPDATE_PREFERENCE: 'UPDATE_PREFERENCE',
  MARK_CHAPTER_COMPLETE: 'MARK_CHAPTER_COMPLETE',
  MARK_CHAPTER_INCOMPLETE: 'MARK_CHAPTER_INCOMPLETE',
  SET_SURVEY_COMPLETED: 'SET_SURVEY_COMPLETED',
  SET_LOADING: 'SET_LOADING',
  RESET_PROFILE: 'RESET_PROFILE',
};

// Reducer function
function userProfileReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_USER_PROFILE:
      return {
        ...state,
        profile: {
          ...state.profile,
          ...action.payload,
        },
        loading: false,
      };
      
    case actionTypes.UPDATE_PREFERENCE:
      return {
        ...state,
        profile: {
          ...state.profile,
          [action.payload.key]: action.payload.value,
        },
      };
      
    case actionTypes.MARK_CHAPTER_COMPLETE:
      if (!state.profile.completedChapters.includes(action.payload.chapterId)) {
        return {
          ...state,
          profile: {
            ...state.profile,
            completedChapters: [...state.profile.completedChapters, action.payload.chapterId],
          },
        };
      }
      return state;
      
    case actionTypes.MARK_CHAPTER_INCOMPLETE:
      return {
        ...state,
        profile: {
          ...state.profile,
          completedChapters: state.profile.completedChapters.filter(
            id => id !== action.payload.chapterId
          ),
        },
      };
      
    case actionTypes.SET_SURVEY_COMPLETED:
      return {
        ...state,
        isSurveyCompleted: action.payload,
      };
      
    case actionTypes.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
      };
      
    case actionTypes.RESET_PROFILE:
      return {
        ...initialState,
        loading: false,
      };
      
    default:
      return state;
  }
}

// Create the context
const UserProfileContext = createContext();

// Provider component
export function UserProfileProvider({ children }) {
  const [state, dispatch] = useReducer(userProfileReducer, initialState);
  
  // Load profile from localStorage on initialization
  React.useEffect(() => {
    const savedProfile = localStorage.getItem('robotics-textbook-user-profile');
    const savedSurveyStatus = localStorage.getItem('robotics-textbook-survey-completed');
    
    if (savedProfile) {
      const profile = JSON.parse(savedProfile);
      dispatch({ type: actionTypes.SET_USER_PROFILE, payload: profile });
    } else {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });
    }
    
    if (savedSurveyStatus) {
      dispatch({ type: actionTypes.SET_SURVEY_COMPLETED, payload: JSON.parse(savedSurveyStatus) });
    }
  }, []);
  
  // Save profile to localStorage whenever it changes
  React.useEffect(() => {
    if (!state.loading) {
      localStorage.setItem('robotics-textbook-user-profile', JSON.stringify(state.profile));
      localStorage.setItem('robotics-textbook-survey-completed', JSON.stringify(state.isSurveyCompleted));
    }
  }, [state.profile, state.isSurveyCompleted, state.loading]);
  
  // Function to update user preference
  const updatePreference = (key, value) => {
    dispatch({
      type: actionTypes.UPDATE_PREFERENCE,
      payload: { key, value }
    });
  };
  
  // Function to mark a chapter as complete
  const markChapterComplete = (chapterId) => {
    dispatch({
      type: actionTypes.MARK_CHAPTER_COMPLETE,
      payload: { chapterId }
    });
  };
  
  // Function to mark a chapter as incomplete
  const markChapterIncomplete = (chapterId) => {
    dispatch({
      type: actionTypes.MARK_CHAPTER_INCOMPLETE,
      payload: { chapterId }
    });
  };
  
  // Function to set survey completion status
  const setSurveyCompleted = (status) => {
    dispatch({
      type: actionTypes.SET_SURVEY_COMPLETED,
      payload: status
    });
  };
  
  // Function to reset profile (for testing purposes)
  const resetProfile = () => {
    dispatch({ type: actionTypes.RESET_PROFILE });
  };
  
  const value = {
    ...state,
    updatePreference,
    markChapterComplete,
    markChapterIncomplete,
    setSurveyCompleted,
    resetProfile,
  };
  
  return (
    <UserProfileContext.Provider value={value}>
      {children}
    </UserProfileContext.Provider>
  );
}

// Custom hook to use the user profile context
export function useUserProfile() {
  const context = useContext(UserProfileContext);
  if (context === undefined) {
    throw new Error('useUserProfile must be used within a UserProfileProvider');
  }
  return context;
}
// src/components/Chatbot.jsx
import React, { useState, useRef, useEffect } from 'react';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your robotics textbook assistant. Ask me anything about the content.", sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Function to scroll to the bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Function to handle sending a message
  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    
    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      // Simulate getting a response from our RAG service
      // In a real implementation, this would call the backend RAG API
      const response = await getRagResponse(inputValue);
      
      // Add bot response
      const botMessage = {
        id: Date.now() + 1,
        text: response,
        sender: 'bot'
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error processing your request. Please try again.",
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Function to get response from RAG service
  const getRagResponse = (query) => {
    return new Promise((resolve) => {
      // Simulate API delay
      setTimeout(() => {
        // In a real implementation, we would:
        // 1. Call window.RagService.search(query)
        // 2. Use the results to generate a contextual response
        // 3. Include citations to source materials
        
        // For this demo, we'll provide a mock implementation
        if (window.RagService) {
          // Perform search using our RAG service
          const results = window.RagService.search(query, 3);
          
          if (results.length > 0 && results[0].similarity > 0.1) {
            // Generate a response based on the search results
            const topResult = results[0];
            const response = `Based on the textbook content, I found relevant information:\n\n"${topResult.content.substring(0, 200)}..."\n\nThis information comes from chapter: ${topResult.docId}\n\n${results.length > 1 ? `I found ${results.length} relevant sections in total.` : ''}`;
            
            resolve(response);
          } else {
            resolve("I couldn't find specific information about that in the textbook. Could you try rephrasing your question or ask about a different topic?");
          }
        } else {
          resolve("The RAG service is not available. This feature will be fully functional when the application is running with the complete implementation.");
        }
      }, 1000); // Simulate network delay
    });
  };
  
  // Function to toggle chatbot open/close
  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };
  
  // Function to clear chat history
  const clearChat = () => {
    setMessages([
      { id: 1, text: "Hello! I'm your robotics textbook assistant. Ask me anything about the content.", sender: 'bot' }
    ]);
  };

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : 'closed'}`}>
      {!isOpen ? (
        <button className="chatbot-toggle" onClick={toggleChatbot}>
          ðŸ¤– AI Assistant
        </button>
      ) : (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="chatbot-title">Robotics Textbook AI</div>
            <div className="chatbot-controls">
              <button className="chatbot-clear" onClick={clearChat} title="Clear chat">
                âŒ«
              </button>
              <button className="chatbot-close" onClick={toggleChatbot} title="Close">
                Ã—
              </button>
            </div>
          </div>
          
          <div className="chatbot-messages">
            {messages.map((message) => (
              <div 
                key={message.id} 
                className={`message ${message.sender}`}
              >
                <div className="message-text">{message.text}</div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="message-text typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <form className="chatbot-input-form" onSubmit={handleSend}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about robotics concepts..."
              disabled={isLoading}
            />
            <button 
              type="submit" 
              disabled={!inputValue.trim() || isLoading}
            >
              Send
            </button>
          </form>
        </div>
      )}
      
      <style jsx>{`
        .chatbot-container {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 1000;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }
        
        .chatbot-toggle {
          background-color: #1a1a1a;
          color: white;
          border: 2px solid #2196F3;
          border-radius: 50%;
          width: 60px;
          height: 60px;
          font-size: 16px;
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
        }

        .chatbot-toggle:hover {
          background-color: #2d2d2d;
          border-color: #1976D2;
          transform: scale(1.05);
        }
        
        .chatbot-window {
          width: 380px;
          height: 500px;
          background-color: white;
          border-radius: 12px;
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          border: 1px solid #e0e0e0;
        }
        
        .chatbot-header {
          background-color: #2196F3;
          color: white;
          padding: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .chatbot-title {
          font-weight: 600;
          font-size: 16px;
        }
        
        .chatbot-controls {
          display: flex;
          gap: 8px;
        }
        
        .chatbot-clear, .chatbot-close {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          font-size: 18px;
          width: 30px;
          height: 30px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .chatbot-clear:hover, .chatbot-close:hover {
          background-color: rgba(255, 255, 255, 0.2);
        }
        
        .chatbot-messages {
          flex: 1;
          padding: 16px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 12px;
          background-color: #f9f9f9;
        }
        
        .message {
          max-width: 80%;
          padding: 10px 14px;
          border-radius: 18px;
          font-size: 14px;
          line-height: 1.4;
        }
        
        .user {
          align-self: flex-end;
          background-color: #2196F3;
          color: white;
          border-bottom-right-radius: 4px;
        }
        
        .bot {
          align-self: flex-start;
          background-color: #e3f2fd;
          color: #333;
          border-bottom-left-radius: 4px;
        }
        
        .typing-indicator {
          display: flex;
          align-items: center;
        }
        
        .typing-indicator span {
          height: 8px;
          width: 8px;
          background-color: #999;
          border-radius: 50%;
          display: inline-block;
          margin: 0 2px;
          animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }
        
        @keyframes typing {
          0%, 60%, 100% { transform: translateY(0); }
          30% { transform: translateY(-5px); }
        }
        
        .chatbot-input-form {
          display: flex;
          padding: 12px;
          background-color: white;
          border-top: 1px solid #e0e0e0;
        }
        
        .chatbot-input-form input {
          flex: 1;
          padding: 12px 16px;
          border: 1px solid #e0e0e0;
          border-radius: 24px;
          font-size: 14px;
          outline: none;
        }
        
        .chatbot-input-form input:focus {
          border-color: #2196F3;
        }
        
        .chatbot-input-form button {
          margin-left: 8px;
          padding: 12px 20px;
          background-color: #2196F3;
          color: white;
          border: none;
          border-radius: 24px;
          cursor: pointer;
        }
        
        .chatbot-input-form button:disabled {
          background-color: #bbdefb;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default Chatbot;
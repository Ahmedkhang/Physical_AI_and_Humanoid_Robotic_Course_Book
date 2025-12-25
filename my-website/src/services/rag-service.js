// // src/services/rag-service.js
// /**
//  * RAG (Retrieval-Augmented Generation) Service that connects to backend API
//  */
// class RagService {
//   constructor(apiBaseUrl = process.env.REACT_APP_RAG_API_URL || 'http://localhost:8000/api/ask') {
//     this.apiBaseUrl = apiBaseUrl;
//   }

//   /**
//    * Queries the backend RAG service with the user's question
//    * @param {string} query - User's question
//    * @param {number} maxResults - Maximum number of results to return
//    * @returns {Promise<Object>} - Response from the backend API
//    */
//   async search(query, maxResults = 5) {
//     try {
//       const response = await fetch(this.apiBaseUrl, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           query,
//           max_results: maxResults
//         })
//       });

//       if (!response.ok) {
//         throw new Error(`RAG service error: ${response.status} ${response.statusText}`);
//       }

//       const data = await response.json();

//       // Handle the response based on your API's format
//       // Assuming your API returns a response with a 'response' field
//       return {
//         content: data.response || data.answer || data.result || data,
//         source: data.source || data.doc_id || 'Unknown'
//       };
//     } catch (error) {
//       console.error('Error querying RAG service:', error);
//       throw error;
//     }
//   }

//   /**
//    * Gets the source document for a chunk
//    * @param {string} chunkId - Chunk identifier
//    * @returns {Object} - Document information
//    */
//   async getChunkSource(chunkId) {
//     // Implementation may vary depending on your backend API design
//     // This is a placeholder since your API might not have a dedicated source endpoint
//     try {
//       // If your API doesn't have a dedicated source endpoint, return null or implement as needed
//       return null;
//     } catch (error) {
//       console.error('Error getting chunk source:', error);
//       return null;
//     }
//   }
// }

// // Create a singleton instance
// const ragService = new RagService();

// // Make it globally available as before
// window.RagService = ragService;

// export default ragService;

// src/services/rag-service.js
// src/services/rag-service.js

class RagService {
  constructor() {
    // Safe way to get env var in Docusaurus/Vite — no import.meta needed
    this.apiBaseUrl = 
      import.meta.env?.VITE_RAG_API_URL ||
      (typeof process !== 'undefined' && process.env?.VITE_RAG_API_URL) ||
      'http://localhost:8000/api/ask';

    this.apiBaseUrl = this.apiBaseUrl.trim();
    console.log('%c RAG Service Loaded → URL:', 'color: green; font-weight: bold', this.apiBaseUrl);
  }

  async search(query) {
    if (!query || query.trim() === '') return { content: 'Please enter a question.', sources: [] };

    try {
      const response = await fetch(this.apiBaseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error ${response.status}: ${errorText || response.statusText}`);
      }

      const data = await response.json();

      return {
        content: data.answer || "No relevant answer found in the book.",
        sources: data.sources || []
      };
    } catch (error) {
      console.error('RAG request failed:', error);
      return {
        content: "Sorry, the textbook assistant is currently unavailable. Make sure the backend is running on port 8000.",
        sources: []
      };
    }
  }
}

// Create singleton and expose globally
const ragService = new RagService();
window.RagService = ragService;

export default ragService;
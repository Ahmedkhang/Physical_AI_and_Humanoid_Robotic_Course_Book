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

// src/services/rag-service.js

// src/services/rag-service.js

class RagService {
  constructor() {
    // Safely read VITE env var — works in build and runtime
    this.apiBaseUrl =
      import.meta.env?.VITE_RAG_API_URL ||
      'https://ahmedur-book-backend.hf.space/api/ask';  // fallback to live URL

    this.apiBaseUrl = this.apiBaseUrl.trim();
  }

  async search(query) {
    if (!query?.trim()) {
      return { content: 'Please ask a question about the course book.', sources: [] };
    }

    try {
      const response = await fetch(this.apiBaseUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() })
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
      }

      const data = await response.json();

      return {
        content: data.answer || 'No relevant information found in the textbook.',
        sources: data.sources || []
      };
    } catch (error) {
      // Safe console.error — only runs in browser
      if (typeof console !== 'undefined') {
        console.error('RAG request failed:', error);
      }
      return {
        content: 'The textbook assistant is temporarily unavailable. Please try again later.',
        sources: []
      };
    }
  }
}

// Create singleton
const ragService = new RagService();

// ONLY assign to window in browser environment
if (typeof window !== 'undefined') {
  window.RagService = ragService;
  // Optional debug log — safe because it's inside browser check
  console.log('%cRAG Service Loaded → URL:', 'color: green; font-weight: bold', ragService.apiBaseUrl);
}

export default ragService;

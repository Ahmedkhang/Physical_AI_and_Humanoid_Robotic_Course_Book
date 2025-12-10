// src/services/rag-service.js
/**
 * RAG (Retrieval-Augmented Generation) Service for the robotics textbook
 * Implements chunking with chunk_size=1024 and overlap=200
 */

class RagService {
  constructor() {
    this.chunkSize = 1024;  // As specified in the requirements
    this.overlap = 200;     // As specified in the requirements
    this.chunks = [];
    this.documents = new Map();
  }

  /**
   * Indexes content for retrieval
   * @param {string} docId - Document identifier
   * @param {string} content - Content to index
   */
  indexDocument(docId, content) {
    // Add document to our collection
    this.documents.set(docId, content);
    
    // Create chunks with specified parameters
    const chunks = this.createChunks(content, docId);
    
    // Add chunks to our collection
    chunks.forEach((chunk, index) => {
      this.chunks.push({
        id: `${docId}-chunk-${index}`,
        docId,
        content: chunk.text,
        start: chunk.start,
        end: chunk.end,
        embedding: this.createEmbedding(chunk.text) // In a real implementation, this would create actual embeddings
      });
    });
    
    console.log(`Indexed document ${docId} with ${chunks.length} chunks`);
  }

  /**
   * Creates chunks with specified chunk_size and overlap
   * @param {string} content - Content to chunk
   * @param {string} docId - Document ID for the chunks
   * @returns {Array} - Array of chunks with text, start, and end positions
   */
  createChunks(content, docId) {
    const chunks = [];
    let start = 0;
    
    while (start < content.length) {
      // Determine the end position for this chunk
      let end = start + this.chunkSize;
      
      // If this is not the last chunk, create an overlap
      if (end < content.length) {
        // Find a good breaking point near the end to avoid cutting sentences
        const breakpoint = this.findBreakpoint(content, end, end - this.overlap);
        end = breakpoint;
      } else {
        end = content.length;
      }
      
      // Extract the chunk text
      const text = content.substring(start, end);
      
      // Add the chunk to our array
      chunks.push({
        text,
        start,
        end,
        docId
      });
      
      // Move the start position, accounting for overlap
      start = end - this.overlap;
      
      // If start is >= content length, we're done
      if (start >= content.length) {
        break;
      }
      
      // Prevent infinite loops in case of unexpected conditions
      if (start >= end) {
        start = end + 1;
      }
    }
    
    return chunks;
  }
  
  /**
   * Finds a good breakpoint for chunking (e.g., at sentence or paragraph boundaries)
   * @param {string} content - Content to find breakpoint in
   * @param {number} fallbackEnd - Fallback end position
   * @param {number} searchStart - Starting position for the search
   * @returns {number} - Position to break at
   */
  findBreakpoint(content, fallbackEnd, searchStart) {
    // Look for a good breaking point near the end of the chunk
    const searchEnd = Math.min(fallbackEnd + 50, content.length); // Look ahead a bit
    const slice = content.slice(searchStart, searchEnd);
    
    // Look for sentence boundaries first
    let lastSentence = slice.lastIndexOf('. ');
    if (lastSentence !== -1) {
      return searchStart + lastSentence + 2; // Include the period and space
    }
    
    // Then look for paragraph breaks
    let lastPara = slice.lastIndexOf('\n\n');
    if (lastPara !== -1) {
      return searchStart + lastPara + 2; // Include both newlines
    }
    
    // Then look for sentence end without space
    let lastSentenceAlt = slice.lastIndexOf('.');
    if (lastSentenceAlt !== -1) {
      return searchStart + lastSentenceAlt + 1;
    }
    
    // Finally, just use the fallback if no good breakpoints found
    return fallbackEnd;
  }
  
  /**
   * Creates a simple embedding representation of text
   * In a real implementation, this would use a proper embedding model
   * @param {string} text - Text to create embedding for
   * @returns {Array} - Embedding array
   */
  createEmbedding(text) {
    // This is a simplified representation - in a real system this would be
    // a high-dimensional vector from a model like SentenceTransformers
    return Array.from({ length: 384 }, (_, i) => {
      // Create a pseudo-embedding based on character codes
      let sum = 0;
      for (let j = 0; j < text.length; j++) {
        sum += text.charCodeAt(j) * (j + 1);
      }
      return Math.sin(sum * (i + 1) / text.length);
    });
  }
  
  /**
   * Searches for relevant chunks based on a query
   * @param {string} query - Query text
   * @returns {Array} - Array of relevant chunks with scores
   */
  search(query, maxResults = 5) {
    // Create embedding for the query
    const queryEmbedding = this.createEmbedding(query);
    
    // Calculate similarity scores for all chunks
    const scoredChunks = this.chunks.map(chunk => {
      const similarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
      return {
        ...chunk,
        similarity
      };
    });
    
    // Sort by similarity score (descending) and return top results
    return scoredChunks
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxResults);
  }
  
  /**
   * Calculates cosine similarity between two vectors
   * @param {Array} vecA - First vector
   * @param {Array} vecB - Second vector
   * @returns {number} - Cosine similarity score
   */
  cosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) {
      throw new Error("Vectors must have the same length");
    }
    
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magnitudeA += vecA[i] ** 2;
      magnitudeB += vecB[i] ** 2;
    }
    
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    
    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0; // If either vector is zero, similarity is 0
    }
    
    return dotProduct / (magnitudeA * magnitudeB);
  }
  
  /**
   * Gets the source document for a chunk
   * @param {string} chunkId - Chunk identifier
   * @returns {Object} - Document information
   */
  getChunkSource(chunkId) {
    const chunk = this.chunks.find(c => c.id === chunkId);
    if (!chunk) return null;
    
    return {
      id: chunk.docId,
      content: this.documents.get(chunk.docId),
      chunkText: chunk.content
    };
  }
}

// Create a singleton instance
const ragService = new RagService();

// Index all textbook content when the service is initialized
// In a real implementation, this would happen when content is added
window.RagService = ragService;

export default ragService;
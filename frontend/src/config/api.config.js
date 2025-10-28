/**
 * API configuration settings.
 * Centralizes all API-related configuration.
 */

export const API_CONFIG = {
  // Base URL from environment or default
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',

  // API endpoints
  ENDPOINTS: {
    HEALTH: '/api/health',
    PROCESS: '/api/process',
  },

  // Request configuration
  TIMEOUT: 600000, // 600 seconds (10 minutes) for file processing - needed for HAC mode with embeddings on large files
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB

  // Response handling
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // ms
}

export default API_CONFIG

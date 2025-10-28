import axios from 'axios'
import API_CONFIG from '../config/api.config'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
})

export const uploadFile = async (
  file,
  useAdaptiveThreshold = false,
  embeddingMode = 'openai-small',
  clusteringMode = 'fixed',
  hacThreshold = 0.42
) => {
  const formData = new FormData()
  formData.append('file', file)

  // Build query parameters
  const params = new URLSearchParams({
    use_adaptive_threshold: useAdaptiveThreshold,
    embedding_mode: embeddingMode,
    clustering_mode: clusteringMode,
  })

  // Only add HAC parameters if in HAC mode
  if (clusteringMode === 'hac') {
    params.append('hac_threshold', hacThreshold)
    params.append('hac_linkage', 'average')
  }

  try {
    const response = await api.post(
      `/api/process?${params.toString()}`,
      formData
    )
    return response.data
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error')
    } else if (error.request) {
      throw new Error('No response from server. Is the backend running?')
    } else {
      throw new Error('Error uploading file')
    }
  }
}

export const healthCheck = async () => {
  try {
    const response = await api.get('/api/health')
    return response.data
  } catch (error) {
    throw new Error('Backend health check failed')
  }
}

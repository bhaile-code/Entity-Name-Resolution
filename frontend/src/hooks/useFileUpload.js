/**
 * Custom hook for file upload logic.
 */

import { useState } from 'react'
import { uploadFile } from '../services/api'
import { validateFile } from '../utils'

/**
 * Hook to handle file upload with validation and state management.
 * @param {Function} onSuccess - Callback when upload succeeds
 * @param {Function} onError - Callback when upload fails
 */
export const useFileUpload = (onSuccess, onError) => {
  const [loading, setLoading] = useState(false)
  const [fileName, setFileName] = useState(null)

  const handleFileUpload = async (file) => {
    // Validate file
    const validation = validateFile(file)
    if (!validation.valid) {
      onError(validation.error)
      return
    }

    setFileName(file.name)
    setLoading(true)

    try {
      const data = await uploadFile(file)
      onSuccess(data)
    } catch (err) {
      onError(err.message || 'Failed to process file')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setLoading(false)
    setFileName(null)
  }

  return {
    loading,
    fileName,
    handleFileUpload,
    reset,
  }
}

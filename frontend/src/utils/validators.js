/**
 * Validation utilities for file uploads and data.
 */

import { FILE_CONSTRAINTS } from '../constants'

/**
 * Validate if file is a CSV.
 * @param {File} file - File to validate
 * @returns {{ valid: boolean, error?: string }}
 */
export const validateFile = (file) => {
  if (!file) {
    return { valid: false, error: 'No file provided' }
  }

  // Check file type by extension
  const fileName = file.name.toLowerCase()
  const hasValidExtension = FILE_CONSTRAINTS.ALLOWED_TYPES.some((ext) =>
    fileName.endsWith(ext)
  )

  if (!hasValidExtension) {
    return {
      valid: false,
      error: `Invalid file type. Please upload a ${FILE_CONSTRAINTS.ALLOWED_TYPES.join(' or ')} file`,
    }
  }

  // Check file size
  if (file.size > FILE_CONSTRAINTS.MAX_SIZE) {
    const sizeMB = (FILE_CONSTRAINTS.MAX_SIZE / (1024 * 1024)).toFixed(0)
    return {
      valid: false,
      error: `File size exceeds ${sizeMB}MB limit`,
    }
  }

  return { valid: true }
}

/**
 * Validate if results data has expected structure.
 * @param {Object} data - Data to validate
 * @returns {boolean}
 */
export const validateResults = (data) => {
  return (
    data &&
    Array.isArray(data.mappings) &&
    data.audit_log &&
    data.summary
  )
}

/**
 * Data formatting utilities.
 */

/**
 * Format confidence score as percentage.
 * @param {number} score - Score between 0 and 1
 * @param {number} decimals - Number of decimal places
 * @returns {string}
 */
export const formatConfidence = (score, decimals = 0) => {
  return `${(score * 100).toFixed(decimals)}%`
}

/**
 * Format timestamp to locale string.
 * @param {string} isoString - ISO timestamp string
 * @returns {string}
 */
export const formatTimestamp = (isoString) => {
  try {
    return new Date(isoString).toLocaleString()
  } catch {
    return isoString
  }
}

/**
 * Format time to locale time string.
 * @param {string} isoString - ISO timestamp string
 * @returns {string}
 */
export const formatTime = (isoString) => {
  try {
    return new Date(isoString).toLocaleTimeString()
  } catch {
    return isoString
  }
}

/**
 * Format number with thousands separator.
 * @param {number} num - Number to format
 * @returns {string}
 */
export const formatNumber = (num) => {
  return num.toLocaleString()
}

/**
 * Format decimal to fixed precision.
 * @param {number} num - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string}
 */
export const formatDecimal = (num, decimals = 1) => {
  return num.toFixed(decimals)
}

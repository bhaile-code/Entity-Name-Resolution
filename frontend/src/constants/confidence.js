/**
 * Constants for confidence score thresholds and display.
 */

export const CONFIDENCE_THRESHOLDS = {
  HIGH: 0.9,
  MEDIUM: 0.7,
  LOW: 0.0,
}

export const CONFIDENCE_LABELS = {
  HIGH: 'High Confidence',
  MEDIUM: 'Medium Confidence',
  LOW: 'Low Confidence',
}

export const CONFIDENCE_COLORS = {
  HIGH: '#d1fae5', // green
  MEDIUM: '#fef3c7', // yellow
  LOW: '#fee2e2', // red
}

/**
 * Get confidence level based on score.
 * @param {number} score - Confidence score between 0 and 1
 * @returns {'HIGH' | 'MEDIUM' | 'LOW'}
 */
export const getConfidenceLevel = (score) => {
  if (score >= CONFIDENCE_THRESHOLDS.HIGH) return 'HIGH'
  if (score >= CONFIDENCE_THRESHOLDS.MEDIUM) return 'MEDIUM'
  return 'LOW'
}

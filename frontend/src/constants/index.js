/**
 * Application constants.
 */

export * from './confidence'

export const APP_NAME = 'Company Name Standardizer'
export const APP_VERSION = '2.0.0'

export const FILE_CONSTRAINTS = {
  ALLOWED_TYPES: ['.csv'],
  MAX_SIZE: 10 * 1024 * 1024, // 10MB
  MIME_TYPES: ['text/csv', 'application/csv'],
}

export const SORT_DIRECTIONS = {
  ASC: 'asc',
  DESC: 'desc',
}

export const TAB_NAMES = {
  MAPPINGS: 'mappings',
  AUDIT: 'audit',
}

/**
 * Custom hook for table sorting logic.
 */

import { useState, useMemo } from 'react'
import { SORT_DIRECTIONS } from '../constants'

/**
 * Hook to handle table sorting.
 * @param {Array} data - Data to sort
 * @param {string} initialField - Initial field to sort by
 * @param {string} initialDirection - Initial sort direction
 */
export const useTableSort = (data, initialField = 'original_name', initialDirection = SORT_DIRECTIONS.ASC) => {
  const [sortField, setSortField] = useState(initialField)
  const [sortDirection, setSortDirection] = useState(initialDirection)

  const handleSort = (field) => {
    if (sortField === field) {
      // Toggle direction if same field
      setSortDirection(
        sortDirection === SORT_DIRECTIONS.ASC ? SORT_DIRECTIONS.DESC : SORT_DIRECTIONS.ASC
      )
    } else {
      // New field, default to ascending
      setSortField(field)
      setSortDirection(SORT_DIRECTIONS.ASC)
    }
  }

  const sortedData = useMemo(() => {
    if (!data || data.length === 0) return []

    return [...data].sort((a, b) => {
      const aVal = a[sortField]
      const bVal = b[sortField]

      if (typeof aVal === 'string') {
        return sortDirection === SORT_DIRECTIONS.ASC
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      } else {
        return sortDirection === SORT_DIRECTIONS.ASC ? aVal - bVal : bVal - aVal
      }
    })
  }, [data, sortField, sortDirection])

  return {
    sortField,
    sortDirection,
    handleSort,
    sortedData,
  }
}

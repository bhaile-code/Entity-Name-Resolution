import { useState } from 'react'
import { downloadCSV } from '../services/export'

function ResultsTable({ mappings }) {
  const [sortField, setSortField] = useState('original_name')
  const [sortDirection, setSortDirection] = useState('asc')
  const [filterText, setFilterText] = useState('')

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const handleDownload = () => {
    downloadCSV(mappings, 'company_name_mappings.csv')
  }

  // Filter mappings
  const filteredMappings = mappings.filter(
    (m) =>
      m.original_name.toLowerCase().includes(filterText.toLowerCase()) ||
      m.canonical_name.toLowerCase().includes(filterText.toLowerCase())
  )

  // Sort mappings
  const sortedMappings = [...filteredMappings].sort((a, b) => {
    const aVal = a[sortField]
    const bVal = b[sortField]

    if (typeof aVal === 'string') {
      return sortDirection === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal)
    } else {
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
    }
  })

  return (
    <div className="results-table-container">
      <div className="table-controls">
        <input
          type="text"
          placeholder="Filter by name..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          className="filter-input"
        />
        <button onClick={handleDownload} className="download-btn">
          Download CSV
        </button>
      </div>

      <div className="table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('original_name')}>
                Original Name {sortField === 'original_name' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('canonical_name')}>
                Canonical Name {sortField === 'canonical_name' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('confidence_score')}>
                Confidence {sortField === 'confidence_score' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('group_id')}>
                Group {sortField === 'group_id' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedMappings.map((mapping, index) => (
              <tr key={index}>
                <td>{mapping.original_name}</td>
                <td>
                  <strong>{mapping.canonical_name}</strong>
                </td>
                <td>
                  <span
                    className={`confidence-badge ${
                      mapping.confidence_score >= 0.9
                        ? 'high'
                        : mapping.confidence_score >= 0.7
                        ? 'medium'
                        : 'low'
                    }`}
                  >
                    {(mapping.confidence_score * 100).toFixed(0)}%
                  </span>
                </td>
                <td>{mapping.group_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-footer">
        Showing {sortedMappings.length} of {mappings.length} mappings
      </div>
    </div>
  )
}

export default ResultsTable

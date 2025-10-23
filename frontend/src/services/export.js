/**
 * Export utilities for downloading results
 */

export const downloadCSV = (mappings, filename) => {
  // Create CSV content
  const headers = ['Original Name', 'Canonical Name', 'Confidence Score', 'Group ID']
  const rows = mappings.map((m) => [
    m.original_name,
    m.canonical_name,
    m.confidence_score.toFixed(3),
    m.group_id,
  ])

  const csvContent = [
    headers.join(','),
    ...rows.map((row) =>
      row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(',')
    ),
  ].join('\n')

  downloadFile(csvContent, filename, 'text/csv')
}

export const downloadJSON = (data, filename) => {
  const jsonContent = JSON.stringify(data, null, 2)
  downloadFile(jsonContent, filename, 'application/json')
}

const downloadFile = (content, filename, mimeType) => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()

  // Cleanup
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

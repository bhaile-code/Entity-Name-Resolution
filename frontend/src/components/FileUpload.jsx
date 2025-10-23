import { useState } from 'react'
import { uploadFile } from '../services/api'

function FileUpload({ onFileProcessed, onError, loading, setLoading }) {
  const [dragActive, setDragActive] = useState(false)
  const [fileName, setFileName] = useState(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = async (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      await handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file) => {
    // Validate file type
    if (!file.name.endsWith('.csv')) {
      onError('Please upload a CSV file')
      return
    }

    setFileName(file.name)
    setLoading(true)

    try {
      const data = await uploadFile(file)
      onFileProcessed(data)
    } catch (err) {
      onError(err.message || 'Failed to process file')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="file-upload-container">
      <form
        className={`file-upload-form ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onSubmit={(e) => e.preventDefault()}
      >
        <input
          type="file"
          id="file-input"
          accept=".csv"
          onChange={handleChange}
          disabled={loading}
        />

        <label htmlFor="file-input" className="file-upload-label">
          {loading ? (
            <>
              <div className="spinner"></div>
              <p>Processing {fileName}...</p>
            </>
          ) : (
            <>
              <div className="upload-icon">📊</div>
              <p className="upload-text">
                Drag and drop your CSV file here, or click to browse
              </p>
              <p className="upload-hint">
                CSV should contain a single column of company names
              </p>
            </>
          )}
        </label>
      </form>

      <div className="instructions">
        <h3>Instructions</h3>
        <ol>
          <li>Prepare a CSV file with company names (one column)</li>
          <li>Upload the file using the area above</li>
          <li>Review the standardized mappings and confidence scores</li>
          <li>Download the results for use in your systems</li>
        </ol>
      </div>
    </div>
  )
}

export default FileUpload

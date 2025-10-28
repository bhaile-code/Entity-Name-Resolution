import { useState } from 'react'
import { uploadFile } from '../services/api'

function FileUpload({ onFileProcessed, onError, loading, setLoading }) {
  const [dragActive, setDragActive] = useState(false)
  const [fileName, setFileName] = useState(null)
  const [thresholdMode, setThresholdMode] = useState('fixed')
  const [embeddingMode, setEmbeddingMode] = useState('openai-small')
  const [hacThreshold, setHacThreshold] = useState(0.42) // Default: 42% distance = 58% similarity
  const [apiError, setApiError] = useState(null)

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
    setApiError(null)

    try {
      const useAdaptive = thresholdMode === 'adaptive'
      const clusteringMode = thresholdMode === 'adaptive' ? 'adaptive_gmm' : (thresholdMode === 'hac' ? 'hac' : 'fixed')
      const data = await uploadFile(file, useAdaptive, embeddingMode, clusteringMode, hacThreshold)
      onFileProcessed(data)
    } catch (err) {
      const errorMsg = err.message || 'Failed to process file'

      // Check if it's an OpenAI API error
      if (errorMsg.includes('OpenAI API unavailable') || errorMsg.includes('OPENAI_API_KEY')) {
        setApiError('OpenAI API is unavailable. Please select "Privacy Mode" to use local embeddings instead.')
      }

      onError(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="file-upload-container">
      <div className="threshold-mode-selector">
        <h3>Clustering Mode</h3>
        <div className="radio-group">
          <label className="radio-option">
            <input
              type="radio"
              name="threshold-mode"
              value="fixed"
              checked={thresholdMode === 'fixed'}
              onChange={(e) => setThresholdMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>Fixed Threshold (Default)</strong>
              <span className="mode-hint">
                Uses preset similarity threshold (85%). Fast and reliable.
              </span>
            </div>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="threshold-mode"
              value="hac"
              checked={thresholdMode === 'hac'}
              onChange={(e) => setThresholdMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>HAC - Hierarchical Clustering (Recommended)</strong>
              <span className="mode-hint">
                Deterministic, reproducible clustering. Configurable threshold below.
              </span>
            </div>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="threshold-mode"
              value="adaptive"
              checked={thresholdMode === 'adaptive'}
              onChange={(e) => setThresholdMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>Adaptive GMM-Based (Experimental)</strong>
              <span className="mode-hint">
                Data-driven thresholds using Gaussian Mixture Model. May be unstable.
              </span>
            </div>
          </label>
        </div>

        {thresholdMode === 'hac' && (
          <div className="hac-threshold-slider">
            <h4>HAC Threshold Configuration</h4>
            <div className="slider-container">
              <label htmlFor="hac-threshold-slider">
                Distance Threshold: {hacThreshold.toFixed(2)}
                <span className="similarity-equivalent">
                  {' '}(Similarity: {((1 - hacThreshold) * 100).toFixed(0)}%)
                </span>
              </label>
              <input
                id="hac-threshold-slider"
                type="range"
                min="0.10"
                max="0.60"
                step="0.01"
                value={hacThreshold}
                onChange={(e) => setHacThreshold(parseFloat(e.target.value))}
                disabled={loading}
                className="threshold-slider"
              />
              <div className="slider-labels">
                <span className="slider-label-left">0.10 (90% similarity - Very Strict)</span>
                <span className="slider-label-center">0.42 (58% - Default)</span>
                <span className="slider-label-right">0.60 (40% - Permissive)</span>
              </div>
              <p className="slider-hint">
                Lower values = stricter grouping (more groups), Higher values = looser grouping (fewer groups)
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="embedding-mode-selector">
        <h3>Embedding Quality</h3>
        <p className="selector-description">
          Choose how to analyze company name similarity. Higher quality modes provide better accuracy.
        </p>

        {apiError && (
          <div className="api-error-banner">
            <span className="error-icon">⚠️</span>
            <span>{apiError}</span>
          </div>
        )}

        <div className="radio-group">
          <label className="radio-option">
            <input
              type="radio"
              name="embedding-mode"
              value="openai-large"
              checked={embeddingMode === 'openai-large'}
              onChange={(e) => setEmbeddingMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>Best Quality</strong>
              <span className="mode-hint">
                OpenAI text-embedding-3-large (~90% accuracy, $0.13/1M tokens)
              </span>
            </div>
          </label>

          <label className="radio-option">
            <input
              type="radio"
              name="embedding-mode"
              value="openai-small"
              checked={embeddingMode === 'openai-small'}
              onChange={(e) => setEmbeddingMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>Balanced (Recommended)</strong>
              <span className="mode-hint">
                OpenAI text-embedding-3-small (~85% accuracy, $0.02/1M tokens)
              </span>
            </div>
          </label>

          <label className="radio-option">
            <input
              type="radio"
              name="embedding-mode"
              value="local"
              checked={embeddingMode === 'local'}
              onChange={(e) => setEmbeddingMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>Privacy Mode</strong>
              <span className="mode-hint">
                Local processing only (~75% accuracy, no API calls, data stays on your machine)
              </span>
            </div>
          </label>

          <label className="radio-option">
            <input
              type="radio"
              name="embedding-mode"
              value="disabled"
              checked={embeddingMode === 'disabled'}
              onChange={(e) => setEmbeddingMode(e.target.value)}
              disabled={loading}
            />
            <div className="radio-label">
              <strong>Disabled (Fuzzy Matching Only)</strong>
              <span className="mode-hint">
                Original algorithm without embeddings (~61% accuracy, fastest)
              </span>
            </div>
          </label>
        </div>
      </div>

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

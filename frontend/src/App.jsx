import { useState } from 'react'
import FileUpload from './components/FileUpload'
import ResultsTable from './components/ResultsTable'
import Summary from './components/Summary'
import AuditLog from './components/AuditLog'
import './App.css'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('mappings') // 'mappings' or 'audit'

  const handleFileProcessed = (data) => {
    setResults(data)
    setError(null)
  }

  const handleError = (errorMessage) => {
    setError(errorMessage)
    setResults(null)
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
    setActiveTab('mappings')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Company Name Standardizer</h1>
        <p className="subtitle">
          Automatically group and normalize similar company names
        </p>
      </header>

      <main className="app-main">
        {!results && (
          <FileUpload
            onFileProcessed={handleFileProcessed}
            onError={handleError}
            loading={loading}
            setLoading={setLoading}
          />
        )}

        {error && (
          <div className="error-message">
            <h3>Error Processing File</h3>
            <p>{error}</p>
            <button onClick={handleReset}>Try Again</button>
          </div>
        )}

        {results && (
          <div className="results-container">
            <Summary summary={results.summary} gmmMetadata={results.gmm_metadata} />

            <div className="tabs">
              <button
                className={`tab ${activeTab === 'mappings' ? 'active' : ''}`}
                onClick={() => setActiveTab('mappings')}
              >
                Mappings ({results.mappings.length})
              </button>
              <button
                className={`tab ${activeTab === 'audit' ? 'active' : ''}`}
                onClick={() => setActiveTab('audit')}
              >
                Audit Log
              </button>
            </div>

            {activeTab === 'mappings' ? (
              <ResultsTable mappings={results.mappings} />
            ) : (
              <AuditLog auditLog={results.audit_log} />
            )}

            <div className="actions">
              <button onClick={handleReset} className="secondary">
                Process Another File
              </button>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>All processing happens locally - no data is sent to external servers</p>
      </footer>
    </div>
  )
}

export default App

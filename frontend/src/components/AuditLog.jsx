import { downloadJSON } from '../services/export'

function AuditLog({ auditLog }) {
  const handleDownload = () => {
    downloadJSON(auditLog, 'audit_log.json')
  }

  return (
    <div className="audit-log-container">
      <div className="audit-log-header">
        <div className="audit-log-info">
          <p>
            <strong>File:</strong> {auditLog.filename}
          </p>
          <p>
            <strong>Processed:</strong>{' '}
            {new Date(auditLog.processed_at).toLocaleString()}
          </p>
          <p>
            <strong>Total Names:</strong> {auditLog.total_names} →{' '}
            <strong>Groups:</strong> {auditLog.total_groups}
          </p>
        </div>
        <button onClick={handleDownload} className="download-btn">
          Download Audit Log
        </button>
      </div>

      <div className="audit-log-entries">
        {auditLog.entries.map((entry, index) => (
          <div key={index} className="audit-entry">
            <div className="audit-entry-header">
              <span className="audit-entry-name">{entry.original_name}</span>
              <span className="audit-entry-time">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="audit-entry-body">
              <p>
                <strong>Mapped to:</strong> {entry.canonical_name}
              </p>
              <p>
                <strong>Confidence:</strong>{' '}
                {(entry.confidence_score * 100).toFixed(1)}%
              </p>
              <p>
                <strong>Group:</strong> {entry.group_id}
              </p>
              <p className="audit-reasoning">{entry.reasoning}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AuditLog

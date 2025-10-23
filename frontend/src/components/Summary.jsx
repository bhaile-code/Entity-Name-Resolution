function Summary({ summary }) {
  return (
    <div className="summary">
      <div className="summary-card">
        <div className="summary-label">Input Names</div>
        <div className="summary-value">{summary.total_input_names}</div>
      </div>
      <div className="summary-card">
        <div className="summary-label">Groups Created</div>
        <div className="summary-value">{summary.total_groups_created}</div>
      </div>
      <div className="summary-card">
        <div className="summary-label">Reduction</div>
        <div className="summary-value">
          {summary.reduction_percentage.toFixed(1)}%
        </div>
      </div>
      <div className="summary-card">
        <div className="summary-label">Avg Group Size</div>
        <div className="summary-value">
          {summary.average_group_size.toFixed(1)}
        </div>
      </div>
      <div className="summary-card">
        <div className="summary-label">Processing Time</div>
        <div className="summary-value">
          {summary.processing_time_seconds.toFixed(2)}s
        </div>
      </div>
    </div>
  )
}

export default Summary

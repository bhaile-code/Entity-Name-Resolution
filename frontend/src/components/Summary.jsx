function Summary({ summary, gmmMetadata }) {
  const thresholdInfo = summary.threshold_info

  return (
    <div className="summary-container">
      {/* Threshold Information */}
      <div className="threshold-info-section">
        <h3>Threshold Method</h3>
        {thresholdInfo.method === 'adaptive_gmm' ? (
          <div className="threshold-adaptive">
            <div className="threshold-badge adaptive">Adaptive GMM-Based</div>
            <div className="threshold-values">
              <div className="threshold-item">
                <span className="threshold-label">T_LOW (Reject):</span>
                <span className="threshold-value">{(thresholdInfo.t_low * 100).toFixed(1)}%</span>
              </div>
              <div className="threshold-item">
                <span className="threshold-label">S_90 (Promotion):</span>
                <span className="threshold-value">{(thresholdInfo.s_90 * 100).toFixed(1)}%</span>
              </div>
              <div className="threshold-item">
                <span className="threshold-label">T_HIGH (Auto-Accept):</span>
                <span className="threshold-value">{(thresholdInfo.t_high * 100).toFixed(1)}%</span>
              </div>
            </div>
            {gmmMetadata && (
              <div className="gmm-stats">
                <h4>GMM Cluster Statistics</h4>
                <div className="gmm-details">
                  <div className="gmm-item">
                    <span className="gmm-label">Cluster Means:</span>
                    <span className="gmm-value">
                      {gmmMetadata.cluster_means.map(m => (m * 100).toFixed(1) + '%').join(', ')}
                    </span>
                  </div>
                  <div className="gmm-item">
                    <span className="gmm-label">Cluster Weights:</span>
                    <span className="gmm-value">
                      {gmmMetadata.cluster_weights.map(w => (w * 100).toFixed(1) + '%').join(', ')}
                    </span>
                  </div>
                  <div className="gmm-item">
                    <span className="gmm-label">Pairs Analyzed:</span>
                    <span className="gmm-value">{gmmMetadata.total_pairs_analyzed}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="threshold-fixed">
            <div className="threshold-badge fixed">Fixed Threshold</div>
            <div className="threshold-value-large">{thresholdInfo.fixed_threshold}%</div>
            {thresholdInfo.fallback_reason && (
              <div className="fallback-warning">
                <strong>Note:</strong> {thresholdInfo.fallback_reason}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Summary Statistics */}
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
    </div>
  )
}

export default Summary

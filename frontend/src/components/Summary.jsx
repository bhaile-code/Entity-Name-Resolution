function Summary({ summary, gmmMetadata }) {
  const thresholdInfo = summary.threshold_info
  const llmMetadata = summary.llm_borderline

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

      {/* LLM Borderline Assessment Statistics */}
      {llmMetadata && llmMetadata.enabled && (
        <div className="llm-metadata-section">
          <h3>🤖 AI Borderline Assessment</h3>
          <div className="llm-overview">
            <div className="llm-badge-info">
              <span className="llm-model-badge">{llmMetadata.llm_model}</span>
              <span className="llm-cost-estimate">Est. Cost: ${llmMetadata.api_cost_estimate.toFixed(4)}</span>
            </div>
            <div className="llm-stats-grid">
              <div className="llm-stat-item">
                <span className="llm-stat-label">Borderline Pairs:</span>
                <span className="llm-stat-value">{llmMetadata.total_borderline_pairs}</span>
              </div>
              <div className="llm-stat-item">
                <span className="llm-stat-label">LLM Assessments:</span>
                <span className="llm-stat-value">{llmMetadata.llm_assessments_made}</span>
              </div>
              <div className="llm-stat-item">
                <span className="llm-stat-label">Cache Hits:</span>
                <span className="llm-stat-value">{llmMetadata.cache_hits}</span>
              </div>
              <div className="llm-stat-item">
                <span className="llm-stat-label">Adjustments Applied:</span>
                <span className="llm-stat-value">{llmMetadata.adjustments_applied}</span>
              </div>
            </div>
          </div>

          {llmMetadata.guardrail_stats && (
            <div className="guardrail-stats">
              <h4>Guardrail Statistics</h4>
              <div className="guardrail-grid">
                <div className="guardrail-item">
                  <span className="guardrail-label">Total Assessments:</span>
                  <span className="guardrail-value">{llmMetadata.guardrail_stats.total_assessments}</span>
                </div>
                <div className="guardrail-item">
                  <span className="guardrail-label">Guardrails Triggered:</span>
                  <span className="guardrail-value guardrail-warning">
                    {llmMetadata.guardrail_stats.guardrails_triggered}
                  </span>
                </div>
                <div className="guardrail-item">
                  <span className="guardrail-label">Same Decisions:</span>
                  <span className="guardrail-value">{llmMetadata.guardrail_stats.same_decisions} 🤖✓</span>
                </div>
                <div className="guardrail-item">
                  <span className="guardrail-label">Different Decisions:</span>
                  <span className="guardrail-value">{llmMetadata.guardrail_stats.different_decisions} 🤖✗</span>
                </div>
                <div className="guardrail-item">
                  <span className="guardrail-label">Unknown Responses:</span>
                  <span className="guardrail-value">{llmMetadata.guardrail_stats.unknown_responses} 🤖❓</span>
                </div>
                <div className="guardrail-item">
                  <span className="guardrail-label">Avg Confidence:</span>
                  <span className="guardrail-value">
                    {(llmMetadata.guardrail_stats.avg_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="guardrail-item">
                  <span className="guardrail-label">Low Confidence Converted:</span>
                  <span className="guardrail-value guardrail-warning">
                    {llmMetadata.guardrail_stats.low_confidence_converted}
                  </span>
                </div>
              </div>
              <div className="guardrail-info">
                <p className="guardrail-note">
                  Guardrails ensure honest LLM responses by flagging low confidence, weak reasoning, and potential hallucinations.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

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

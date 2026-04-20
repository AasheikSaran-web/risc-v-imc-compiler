import { AnalysisResult } from '../compiler/types';

interface AnalysisPanelProps {
  analysis?: AnalysisResult;
}

export function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  if (!analysis) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Compile a kernel to see analysis results.
      </div>
    );
  }

  const { metrics, divergence, coalescing } = analysis;

  const memScore = metrics.memoryInstructions > 0
    ? Math.round((1 - (coalescing.filter(c => c.accessPattern !== 'coalesced').length / Math.max(coalescing.length, 1))) * 100)
    : 100;
  const divScore = divergence.length === 0 ? 100 : Math.round((1 - divergence.length / Math.max(metrics.branchInstructions, 1)) * 100);
  const regScore = Math.round((1 - metrics.registersUsed / 20) * 100);
  const imcBonus = metrics.imcOperations > 0 ? Math.min(100, metrics.imcOperations * 25) : 0;
  const overallScore = Math.min(100, Math.round((memScore + divScore + regScore) / 3 + imcBonus * 0.2));

  return (
    <div style={{ height: '100%', overflow: 'auto', padding: '8px', fontSize: '12px', fontFamily: 'monospace' }}>

      {/* Performance Score */}
      <div style={{ marginBottom: '12px', padding: '8px', background: '#1a1a2e', borderRadius: '6px', border: '1px solid #333' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ color: '#888', fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            IMC Performance Score
          </span>
          <span style={{
            fontSize: '20px',
            fontWeight: 700,
            color: overallScore >= 80 ? '#4ec9b0' : overallScore >= 50 ? '#e5c07b' : '#e06c75',
          }}>
            {overallScore}
          </span>
        </div>
        <ScoreBar label="Memory Efficiency" score={memScore} />
        <ScoreBar label="Branch Uniformity" score={divScore} />
        <ScoreBar label="Register Pressure" score={regScore} />
        {metrics.imcOperations > 0 && (
          <ScoreBar label="Crossbar Utilization" score={imcBonus} />
        )}
      </div>

      {/* IMC Crossbar Section */}
      {metrics.imcOperations > 0 && (
        <div style={{ marginBottom: '12px', padding: '8px', background: '#0d1f2d', borderRadius: '6px', border: '1px solid #1a4a6a' }}>
          <SectionHeader title="Memristor Crossbar (IMC)" color="#4ec9b0" />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '6px' }}>
            <MetricCard label="MVM Ops" value={metrics.imcOperations} color="#4ec9b0" />
            <MetricCard label="Cycles Saved" value={metrics.crossbarCyclesSaved} color="#98c379" />
          </div>
          <div style={{
            padding: '6px 8px',
            background: '#1a2a3a',
            borderRadius: '4px',
            color: '#4ec9b0',
            fontSize: '10px',
            lineHeight: '1.5',
          }}>
            Each <strong>MVM</strong> computes a 16×16 matrix-vector product in a single analog step
            (≈1 cycle) by exploiting Kirchhoff's current law across memristor conductances.
            Equivalent to <strong>{metrics.crossbarCyclesSaved} scalar MAC operations</strong> avoided.
          </div>
        </div>
      )}

      {/* Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '12px' }}>
        <MetricCard label="Instructions" value={metrics.totalInstructions} color="#61afef" />
        <MetricCard label="Registers" value={`${metrics.registersUsed}/20`} color="#c586c0" />
        <MetricCard label="Compute Ops" value={metrics.computeInstructions} color="#98c379" />
        <MetricCard label="Memory Ops" value={metrics.memoryInstructions} color="#e5c07b" />
        <MetricCard label="Branches" value={metrics.branchInstructions} color="#e06c75" />
        <MetricCard label="Barriers" value={metrics.barrierCount} color="#d19a66" />
        {(metrics.crossbarWriteOps ?? 0) > 0 && (
          <MetricCard label="CSET Writes" value={metrics.crossbarWriteOps} color="#4ec9b0" />
        )}
        {metrics.sharedMemoryBytes > 0 && (
          <MetricCard label="Scratchpad" value={`${metrics.sharedMemoryBytes}B`} color="#4ec9b0" />
        )}
        <MetricCard
          label="Compute/Memory"
          value={metrics.computeToMemoryRatio.toFixed(1)}
          color={metrics.computeToMemoryRatio >= 2 ? '#98c379' : '#e5c07b'}
        />
      </div>

      {/* Divergence Analysis */}
      {divergence.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <SectionHeader title="Thread Divergence" color="#e06c75" />
          {divergence.map((d, i) => (
            <div key={i} style={{
              padding: '6px 8px', background: '#1a1a2e', border: '1px solid #3a2020',
              borderRadius: '4px', marginBottom: '4px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#e06c75' }}>addr {d.instructionAddr}</span>
                <span style={{ padding: '2px 6px', background: '#3a2020', color: '#e06c75', borderRadius: '3px', fontSize: '10px' }}>
                  DIVERGENT
                </span>
              </div>
              <div style={{ color: '#888', fontSize: '11px', marginTop: '2px' }}>{d.description}</div>
            </div>
          ))}
        </div>
      )}

      {/* Memory Access Patterns */}
      {coalescing.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <SectionHeader title="Memory Access Patterns" color="#61afef" />
          {coalescing.map((c, i) => (
            <div key={i} style={{
              padding: '6px 8px', background: '#1a1a2e',
              border: `1px solid ${c.accessPattern === 'coalesced' ? '#2a3a2a' : c.accessPattern === 'strided' ? '#3a3a20' : '#3a2020'}`,
              borderRadius: '4px', marginBottom: '4px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#9cdcfe' }}>addr {c.instructionAddr}</span>
                <span style={{
                  padding: '2px 6px',
                  background: c.accessPattern === 'coalesced' ? '#2a3a2a' : c.accessPattern === 'strided' ? '#3a3a20' : '#3a2020',
                  color: c.accessPattern === 'coalesced' ? '#98c379' : c.accessPattern === 'strided' ? '#e5c07b' : '#e06c75',
                  borderRadius: '3px', fontSize: '10px', fontWeight: 700,
                }}>
                  {c.accessPattern.toUpperCase()}
                </span>
              </div>
              <div style={{ color: '#888', fontSize: '11px', marginTop: '2px' }}>{c.description}</div>
              {c.transactionsNeeded !== 1 && (
                <div style={{ color: '#666', fontSize: '10px', marginTop: '1px' }}>
                  Transactions: {c.transactionsNeeded}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Workload Balance */}
      <div style={{ marginBottom: '12px' }}>
        <SectionHeader title="Workload Balance" color="#98c379" />
        <div style={{ padding: '8px', background: '#1a1a2e', borderRadius: '4px', border: '1px solid #333' }}>
          <div style={{ display: 'flex', height: '20px', borderRadius: '3px', overflow: 'hidden' }}>
            {metrics.imcOperations > 0 && (
              <div style={{
                width: `${metrics.imcOperations / Math.max(metrics.totalInstructions, 1) * 100}%`,
                background: '#4ec9b0',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '9px', color: '#000', fontWeight: 700,
              }}>
                IMC
              </div>
            )}
            <div style={{
              width: `${metrics.computeInstructions / Math.max(metrics.totalInstructions, 1) * 100}%`,
              background: '#98c379',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '9px', color: '#000', fontWeight: 700,
            }}>
              Compute
            </div>
            <div style={{
              flex: 1, background: '#61afef',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '9px', color: '#000', fontWeight: 700,
            }}>
              Memory
            </div>
          </div>
          <div style={{ color: '#888', fontSize: '10px', marginTop: '4px', textAlign: 'center' }}>
            {metrics.imcOperations > 0
              ? `${metrics.imcOperations} crossbar MVM ops offload heavy matrix compute to analog`
              : metrics.computeToMemoryRatio >= 2
                ? 'Compute-bound — good utilization'
                : metrics.computeToMemoryRatio >= 1
                  ? 'Balanced workload'
                  : 'Memory-bound — consider crossbar MVM or scratchpad reuse'}
          </div>
        </div>
      </div>

      {/* Estimated cycles */}
      <div style={{
        padding: '8px', background: '#1a1a2e', borderRadius: '4px', border: '1px solid #333',
        color: '#888', fontSize: '11px', textAlign: 'center',
      }}>
        Est. <strong style={{ color: '#4ec9b0' }}>{metrics.estimatedCycles}</strong> cycles
        ({metrics.totalInstructions} inst × 5-stage RISC-V pipeline
        {metrics.imcOperations > 0 ? `, ${metrics.imcOperations} IMC single-cycle` : ''})
      </div>
    </div>
  );
}

function ScoreBar({ label, score }: { label: string; score: number }) {
  const color = score >= 80 ? '#4ec9b0' : score >= 50 ? '#e5c07b' : '#e06c75';
  return (
    <div style={{ marginBottom: '4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
        <span style={{ color: '#888', fontSize: '10px' }}>{label}</span>
        <span style={{ color, fontSize: '10px', fontWeight: 700 }}>{score}%</span>
      </div>
      <div style={{ height: '4px', background: '#2a2a3a', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{ width: `${score}%`, height: '100%', background: color, borderRadius: '2px', transition: 'width 0.3s' }} />
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div style={{ padding: '6px 8px', background: '#1a1a2e', borderRadius: '4px', border: '1px solid #333' }}>
      <div style={{ color: '#666', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        {label}
      </div>
      <div style={{ color, fontSize: '16px', fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function SectionHeader({ title, color }: { title: string; color: string }) {
  return (
    <div style={{
      fontSize: '11px', color, fontWeight: 700, textTransform: 'uppercase',
      letterSpacing: '0.5px', marginBottom: '6px', paddingBottom: '4px',
      borderBottom: `1px solid ${color}33`,
    }}>
      {title}
    </div>
  );
}

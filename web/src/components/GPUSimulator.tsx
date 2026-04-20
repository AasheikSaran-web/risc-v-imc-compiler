import { useState, useEffect, useCallback, useRef } from 'react';
import { TinyGPUSim } from '../simulator/TinyGPUSim';
import {
  Instruction, SimulationState, PipelineStage, CrossbarState,
  MEMRISTOR_PHYSICS, MemristorWriteEvent,
} from '../compiler/types';

interface IMCSimulatorProps {
  instructions: Instruction[];
  initialMemory: number[];
  numBlocks: number;
  threadsPerBlock: number;
  onCycleChange?: (state: SimulationState) => void;
}

const STAGE_COLORS: Record<string, string> = {
  [PipelineStage.FETCH]:     '#e06c75',
  [PipelineStage.DECODE]:    '#d19a66',
  [PipelineStage.EXECUTE]:   '#61afef',
  [PipelineStage.MEMORY]:    '#98c379',
  [PipelineStage.WRITEBACK]: '#c678dd',
  [PipelineStage.BARRIER]:   '#4ec9b0',
  [PipelineStage.DONE]:      '#555',
};

const RV_ABI = [
  'zero','ra','sp','gp','tp',
  't0','t1','t2','s0','s1',
  'a0','a1','a2','a3','a4','a5','a6','a7',
  's2','s3','s4','s5','s6','s7','s8',
  'tid','bid','bdm',
  't3','t4','t5','t6',
];

const { G_MIN_US, G_MAX_US, POT_V_MV, POT_NS, DEP_V_MV, DEP_NS } = MEMRISTOR_PHYSICS;

// ── Main component ────────────────────────────────────────────────────────

export function GPUSimulator({
  instructions, initialMemory, numBlocks, threadsPerBlock, onCycleChange,
}: IMCSimulatorProps) {
  const [history, setHistory]           = useState<SimulationState[]>([]);
  const [currentStep, setCurrentStep]   = useState(0);
  const [isPlaying, setIsPlaying]       = useState(false);
  const [speed, setSpeed]               = useState(500);
  const [selectedThread, setSelectedThread] = useState<number | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (instructions.length === 0) return;
    const sim = new TinyGPUSim(instructions, initialMemory, numBlocks, threadsPerBlock);
    setHistory(sim.runToEnd(5000));
    setCurrentStep(0);
    setIsPlaying(false);
    setSelectedThread(null);
  }, [instructions, initialMemory, numBlocks, threadsPerBlock]);

  const state = history[currentStep];

  useEffect(() => {
    if (state && onCycleChange) onCycleChange(state);
  }, [currentStep, state, onCycleChange]);

  useEffect(() => {
    if (isPlaying && history.length > 0) {
      intervalRef.current = window.setInterval(() => {
        setCurrentStep((p) => {
          if (p >= history.length - 1) { setIsPlaying(false); return p; }
          return p + 1;
        });
      }, speed);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isPlaying, speed, history.length]);

  const stepForward  = useCallback(() => setCurrentStep((p) => Math.min(p + 1, history.length - 1)), [history.length]);
  const stepBackward = useCallback(() => setCurrentStep((p) => Math.max(p - 1, 0)), []);
  const reset        = useCallback(() => { setCurrentStep(0); setIsPlaying(false); }, []);
  const jumpToEnd    = useCallback(() => { setCurrentStep(history.length - 1); setIsPlaying(false); }, [history.length]);

  if (!state || instructions.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Compile a kernel to start the IMC simulator.
      </div>
    );
  }

  const selectedThreadState = selectedThread !== null
    ? state.threads.find((t) => t.threadId === selectedThread) ?? null
    : null;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Controls */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '6px', padding: '8px',
        background: '#111', borderRadius: '4px', flexShrink: 0, flexWrap: 'wrap',
      }}>
        <button onClick={reset}         style={btnStyle} title="Reset">{'\u23EE'}</button>
        <button onClick={stepBackward}  style={btnStyle} title="Step Back">{'\u23EA'}</button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={{ ...btnStyle, background: isPlaying ? '#e06c75' : '#2d5a3d', width: '48px' }}
        >
          {isPlaying ? '\u23F8' : '\u25B6'}
        </button>
        <button onClick={stepForward}   style={btnStyle} title="Step Forward">{'\u23E9'}</button>
        <button onClick={jumpToEnd}     style={btnStyle} title="Jump to End">{'\u23ED'}</button>
        <div style={{ flex: 1 }} />
        <label style={{ fontSize: '11px', color: '#888' }}>
          Speed:
          <input
            type="range" min={50} max={1000} step={50} value={1050 - speed}
            onChange={(e) => setSpeed(1050 - parseInt(e.target.value))}
            style={{ width: '60px', marginLeft: '4px', verticalAlign: 'middle' }}
          />
        </label>
        <span style={{ fontSize: '11px', color: '#4ec9b0', fontFamily: 'monospace' }}>
          Cycle {state.cycle} / {history.length - 1}
        </span>
      </div>

      {/* Scrubber */}
      <input
        type="range" min={0} max={history.length - 1} value={currentStep}
        onChange={(e) => { setCurrentStep(parseInt(e.target.value)); setIsPlaying(false); }}
        style={{ width: '100%', flexShrink: 0 }}
      />

      {/* Status bar */}
      <div style={{ fontSize: '11px', color: '#888', padding: '0 4px', flexShrink: 0,
        display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span>Block {state.currentBlock} / {state.totalBlocks}</span>
        {state.currentBlock >= state.totalBlocks && <span style={{ color: '#4ec9b0' }}>DONE</span>}
        {state.threads.some((t) => t.divergent) && (
          <span style={{ color: '#e06c75', padding: '1px 6px', background: '#3a2020',
            borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>DIVERGENT</span>
        )}
        {state.threads.some((t) => t.stage === PipelineStage.BARRIER) && (
          <span style={{ color: '#4ec9b0', padding: '1px 6px', background: '#1a3a3a',
            borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>FENCE/SYNC</span>
        )}
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {/* Thread cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${Math.min(threadsPerBlock, 4)}, 1fr)`,
          gap: '6px', padding: '4px',
        }}>
          {state.threads.map((thread) => (
            <ThreadCard
              key={`${thread.blockId}-${thread.threadId}`}
              thread={thread}
              selected={selectedThread === thread.threadId}
              onClick={() => setSelectedThread(
                selectedThread === thread.threadId ? null : thread.threadId
              )}
            />
          ))}
        </div>

        {/* Selected thread: register file + crossbar data-flow */}
        {selectedThreadState && (
          <div style={{ margin: '8px 4px', padding: '8px', background: '#1a1a2e',
            border: '1px solid #4ec9b0', borderRadius: '6px' }}>
            <div style={{ fontSize: '11px', color: '#4ec9b0', fontWeight: 700, marginBottom: '6px' }}>
              RISC-V Core T{selectedThreadState.threadId} — Register File (x0–x31)
              <span style={{ color: '#555', fontWeight: 400, marginLeft: '8px' }}>
                → each write programs crossbar(reg, col=0) via pot/dep pulses
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '4px', fontSize: '10px', fontFamily: 'monospace' }}>
              {selectedThreadState.registers.map((val, r) => {
                const isSpecial = r === 25 || r === 26 || r === 27;
                const isNonzero = val !== 0 && r !== 0;
                // Compute conductance from crossbar col 0
                const G = state.crossbar.conductances[r]?.[0] ?? G_MIN_US;
                const gNorm = (G - G_MIN_US) / (G_MAX_US - G_MIN_US);
                const borderColor = isNonzero
                  ? `rgba(78,201,176,${0.2 + gNorm * 0.6})`
                  : 'transparent';
                return (
                  <div
                    key={r}
                    title={`x${r} (${RV_ABI[r] ?? '?'}) = ${val} | G = ${G.toFixed(1)} µS | col=0`}
                    style={{
                      padding: '3px 4px', borderRadius: '3px',
                      background: isSpecial ? '#2a2a4a' : isNonzero ? '#1a2a1a' : '#111',
                      border: `1px solid ${borderColor}`,
                      color: isSpecial ? '#c586c0' : isNonzero ? '#98c379' : '#444',
                    }}
                  >
                    <span style={{ color: '#666' }}>{RV_ABI[r] ?? `x${r}`}</span>
                    {' '}{val}
                    {isNonzero && (
                      <div style={{ color: '#4ec9b0', fontSize: '8px', marginTop: '1px' }}>
                        {G.toFixed(0)}µS
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: '4px', fontSize: '10px', color: '#888' }}>
              PC: <span style={{ color: '#b5cea8' }}>{selectedThreadState.pc}</span>
              {selectedThreadState.currentInstruction && (
                <> {' | '}<span style={{ color: '#c586c0' }}>{selectedThreadState.currentInstruction}</span></>
              )}
            </div>
          </div>
        )}

        {/* Crossbar visualization */}
        <CrossbarVisualization crossbar={state.crossbar} />

        {/* Global memory */}
        <div style={{ marginTop: '8px', padding: '4px' }}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Global Memory (256 bytes)
          </div>
          <MemoryHeatmap memory={state.memory} />
        </div>

        {/* Scratchpad memory */}
        {state.sharedMemory.some((v) => v !== 0) && (
          <div style={{ marginTop: '8px', padding: '4px' }}>
            <div style={{ fontSize: '11px', color: '#4ec9b0', marginBottom: '4px' }}>
              Scratchpad Memory (64 bytes)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1px' }}>
              {state.sharedMemory.slice(0, 32).map((val, i) => (
                <div key={i} title={`[S${i}] = ${val}`} style={{
                  width: '20px', height: '18px', display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                  fontSize: '9px', fontFamily: 'monospace',
                  background: val > 0
                    ? `#4ec9b0${Math.min(Math.floor(val / 255 * 200) + 55, 255).toString(16).padStart(2, '0')}`
                    : '#1a1a2e',
                  color: val > 0 ? '#fff' : '#333', borderRadius: '2px', border: '1px solid #222',
                }}>
                  {val}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Crossbar visualization ────────────────────────────────────────────────

function CrossbarVisualization({ crossbar }: { crossbar: CrossbarState }) {
  const gRange = G_MAX_US - G_MIN_US;
  const norm   = (g: number) => Math.max(0, Math.min(1, (g - G_MIN_US) / gRange));

  return (
    <div style={{ margin: '8px 4px' }}>
      {/* ── Section header ── */}
      <div style={{ fontSize: '11px', color: '#4ec9b0', fontWeight: 700,
        marginBottom: '6px', paddingBottom: '3px', borderBottom: '1px solid #4ec9b033' }}>
        Memristor Crossbar — 64×64 [{MEMRISTOR_PHYSICS.MATERIAL}]
        <span style={{ color: '#555', fontWeight: 400, marginLeft: '8px', fontSize: '10px' }}>
          {MEMRISTOR_PHYSICS.G_MIN_US}–{MEMRISTOR_PHYSICS.G_MAX_US} µS · {MEMRISTOR_PHYSICS.LEVELS.toLocaleString()} levels · {MEMRISTOR_PHYSICS.PAPER_REF}
        </span>
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        {/* ── Register-file panel (col 0) ── */}
        <div style={{ flex: '1 1 180px' }}>
          <div style={{ fontSize: '10px', color: '#9cdcfe', marginBottom: '4px', fontWeight: 600 }}>
            Register File ← col 0
            <span style={{ color: '#555', fontWeight: 400, marginLeft: '6px' }}>
              xr → cell(r, 0)
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '2px' }}>
            {Array.from({ length: 32 }, (_, r) => {
              const G    = crossbar.conductances[r]?.[0] ?? G_MIN_US;
              const n    = norm(G);
              const val  = Math.round(n * 255);
              const isActive = G > G_MIN_US + MEMRISTOR_PHYSICS.G_STEP_US;
              return (
                <div
                  key={r}
                  title={`x${r} (${RV_ABI[r]})\nG = ${G.toFixed(2)} µS\nvalue ≈ ${val}\ncell (${r}, 0)`}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '4px',
                    padding: '2px 4px', borderRadius: '3px',
                    background: isActive ? `rgba(78,201,176,${0.08 + n * 0.25})` : '#111',
                    border: `1px solid ${isActive ? `rgba(78,201,176,${0.15 + n*0.5})` : '#1a1a2e'}`,
                    fontSize: '9px', fontFamily: 'monospace',
                  }}
                >
                  <div style={{
                    width: '18px', height: '6px', borderRadius: '2px', flexShrink: 0,
                    background: `linear-gradient(to right, #1a1a2e ${Math.round((1-n)*100)}%, #4ec9b0)`,
                    border: '1px solid #333',
                  }} />
                  <span style={{ color: '#666', fontSize: '8px' }}>{RV_ABI[r] ?? `x${r}`}</span>
                  {isActive && (
                    <span style={{ color: '#4ec9b0', marginLeft: 'auto' }}>
                      {G >= 1000 ? `${(G/1000).toFixed(1)}mS` : `${G.toFixed(0)}µS`}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          {/* Conductance scale legend */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px', fontSize: '9px', color: '#555' }}>
            <span>G_MIN</span>
            <div style={{ flex: 1, height: '4px',
              background: 'linear-gradient(to right, #1a1a2e, #4ec9b0)', borderRadius: '2px' }} />
            <span>G_MAX</span>
          </div>
          <div style={{ fontSize: '9px', color: '#444', marginTop: '2px' }}>
            {G_MIN_US} µS → {G_MAX_US/1000} mS
          </div>
        </div>

        {/* ── Weight matrix panel (cols 1-8, rows 0-7) ── */}
        <div>
          <div style={{ fontSize: '10px', color: '#d19a66', marginBottom: '4px', fontWeight: 600 }}>
            Weights ← cols 1–8
            <span style={{ color: '#555', fontWeight: 400, marginLeft: '6px' }}>CSET/MVM region</span>
          </div>
          {/* Column headers */}
          <div style={{ display: 'flex', gap: '1px', marginBottom: '1px', paddingLeft: '18px' }}>
            {Array.from({ length: 8 }, (_, c) => (
              <div key={c} style={{ width: '22px', fontSize: '8px', color: '#555', textAlign: 'center' }}>
                c{c+1}
              </div>
            ))}
          </div>
          {Array.from({ length: 8 }, (_, r) => (
            <div key={r} style={{ display: 'flex', gap: '1px', marginBottom: '1px', alignItems: 'center' }}>
              <div style={{ width: '18px', fontSize: '8px', color: '#555', textAlign: 'right', paddingRight: '2px' }}>
                r{r}
              </div>
              {Array.from({ length: 8 }, (_, c) => {
                const G = crossbar.conductances[r]?.[c + 1] ?? G_MIN_US;
                const n = norm(G);
                const isActive = G > G_MIN_US + MEMRISTOR_PHYSICS.G_STEP_US;
                return (
                  <div
                    key={c}
                    title={`W[${r}][${c+1}] = ${G.toFixed(1)} µS`}
                    style={{
                      width: '22px', height: '22px', borderRadius: '2px',
                      background: isActive
                        ? `rgba(209,154,102,${0.15 + n * 0.85})`
                        : '#1a1a1a',
                      border: `1px solid ${isActive ? '#d19a6644' : '#222'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '7px', color: n > 0.5 ? '#000' : '#666',
                    }}
                  >
                    {isActive ? Math.round(n * 99) : ''}
                  </div>
                );
              })}
            </div>
          ))}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginTop: '3px', fontSize: '9px', color: '#555' }}>
            <span>low</span>
            <div style={{ width: '40px', height: '4px',
              background: 'linear-gradient(to right, #1a1a1a, #d19a66)', borderRadius: '2px' }} />
            <span>high G</span>
          </div>
          {crossbar.lastMVMResult.some((v) => v !== 0) && (
            <div style={{ marginTop: '6px', fontSize: '9px', color: '#d19a66' }}>
              Last MVM: [{crossbar.lastMVMResult.slice(0, 4).join(', ')}, …]
            </div>
          )}
        </div>
      </div>

      {/* ── Write event log ── */}
      {crossbar.writeEvents.length > 0 && (
        <WriteEventLog events={crossbar.writeEvents} />
      )}
    </div>
  );
}

// ── Write event log ───────────────────────────────────────────────────────

function WriteEventLog({ events }: { events: MemristorWriteEvent[] }) {
  const recent = events.slice(-6).reverse();

  return (
    <div style={{ marginTop: '8px' }}>
      <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px', fontWeight: 600 }}>
        Memristor Write Log
        <span style={{ color: '#555', fontWeight: 400, marginLeft: '6px' }}>
          {MEMRISTOR_PHYSICS.POT_V_MV}mV/{MEMRISTOR_PHYSICS.POT_NS}ns (pot) · {MEMRISTOR_PHYSICS.DEP_V_MV}mV/{MEMRISTOR_PHYSICS.DEP_NS}ns (dep)
        </span>
      </div>
      {recent.map((ev, i) => {
        const isReg   = ev.col === 0;
        const isWeight = ev.col >= 1;
        const pulseColor = ev.pulseType === 'pot' ? '#98c379' : ev.pulseType === 'dep' ? '#e06c75' : '#888';
        const dG = ev.newG_us - ev.prevG_us;
        const arrow = isReg ? '→ reg-file' : isWeight ? '→ weight' : '→ cell';
        return (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            padding: '3px 6px', marginBottom: '2px',
            background: i === 0 ? '#1a2a1a' : '#111',
            borderRadius: '3px',
            border: `1px solid ${i === 0 ? '#2a3a2a' : '#1a1a2e'}`,
            fontSize: '9px', fontFamily: 'monospace',
          }}>
            {/* Cycle */}
            <span style={{ color: '#555', width: '28px' }}>c{ev.cycle}</span>
            {/* Cell address */}
            <span style={{ color: '#9cdcfe' }}>
              ({ev.row},{ev.col})
            </span>
            {/* Register name or weight label */}
            <span style={{ color: '#4ec9b0', width: '36px' }}>
              {ev.registerName ?? `W[${ev.row}][${ev.col}]`}
            </span>
            <span style={{ color: '#555' }}>{arrow}</span>
            {/* Conductance change */}
            <span style={{ color: '#888' }}>
              {ev.prevG_us.toFixed(0)}→
              <span style={{ color: '#b5cea8' }}>{ev.newG_us.toFixed(0)}</span>µS
            </span>
            {/* Pulse info */}
            <span style={{ color: pulseColor, marginLeft: 'auto' }}>
              {ev.pulseType === 'hold' ? 'no-op' : (
                <>
                  {ev.pulseCount}× {ev.pulseType === 'pot' ? `↑${POT_V_MV}mV/${POT_NS}ns` : `↓${DEP_V_MV}mV/${DEP_NS}ns`}
                </>
              )}
            </span>
            {/* Delta G */}
            {Math.abs(dG) > MEMRISTOR_PHYSICS.G_STEP_US && (
              <span style={{ color: dG > 0 ? '#98c379' : '#e06c75', fontSize: '8px' }}>
                {dG > 0 ? '+' : ''}{dG.toFixed(0)}µS
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Thread card ───────────────────────────────────────────────────────────

function ThreadCard({
  thread, selected, onClick,
}: {
  thread: import('../compiler/types').ThreadState;
  selected: boolean;
  onClick: () => void;
}) {
  const stageColor = STAGE_COLORS[thread.stage] || '#555';
  const allocRegs  = thread.registers.slice(5, 13);

  return (
    <div onClick={onClick} style={{
      background: '#1a1a2e',
      border: `1px solid ${selected ? '#4ec9b0' : thread.divergent ? '#e06c75' : thread.done ? '#333' : stageColor}`,
      borderRadius: '6px', padding: '8px', fontSize: '11px', fontFamily: 'monospace',
      opacity: thread.done ? 0.5 : 1, transition: 'all 0.15s', cursor: 'pointer',
      boxShadow: selected ? '0 0 8px #4ec9b044' : thread.divergent ? '0 0 4px #e06c7533' : 'none',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ color: '#9cdcfe' }}>T{thread.threadId}</span>
        <div style={{ display: 'flex', gap: '2px' }}>
          {thread.divergent && (
            <span style={{ background: '#3a2020', color: '#e06c75', padding: '1px 4px',
              borderRadius: '3px', fontSize: '8px', fontWeight: 700 }}>DIV</span>
          )}
          <span style={{ background: stageColor, color: '#fff', padding: '1px 6px',
            borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>{thread.stage}</span>
        </div>
      </div>
      <div style={{ color: '#888', marginBottom: '4px' }}>
        PC: <span style={{ color: '#b5cea8' }}>{thread.pc}</span>
        {thread.currentInstruction && (
          <span style={{ color: '#c586c0', marginLeft: '8px' }}>{thread.currentInstruction}</span>
        )}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
        {allocRegs.map((val, i) => (
          <span key={i} style={{
            padding: '1px 3px', background: val !== 0 ? '#2a2a4a' : 'transparent',
            borderRadius: '2px', color: val !== 0 ? '#e0e0e0' : '#444', fontSize: '9px',
          }} title={`${RV_ABI[5 + i]}=${val}`}>
            {val !== 0 ? val : '\u00B7'}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Memory heatmap ────────────────────────────────────────────────────────

function MemoryHeatmap({ memory }: { memory: number[] }) {
  const regions = [
    { name: 'A (0–63)',    start: 0,   end: 64,  color: '#e06c75' },
    { name: 'B (64–127)', start: 64,  end: 128, color: '#61afef' },
    { name: 'C (128–191)',start: 128, end: 192, color: '#98c379' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {regions.map((region) => (
        <div key={region.name}>
          <div style={{ fontSize: '10px', color: region.color, marginBottom: '2px' }}>{region.name}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1px' }}>
            {memory.slice(region.start, Math.min(region.end, region.start + 16)).map((val, i) => (
              <div key={i} title={`[${region.start + i}] = ${val}`} style={{
                width: '20px', height: '18px', display: 'flex',
                alignItems: 'center', justifyContent: 'center',
                fontSize: '9px', fontFamily: 'monospace',
                background: val > 0
                  ? `${region.color}${Math.min(Math.floor(val / 255 * 200) + 55, 255).toString(16).padStart(2, '0')}`
                  : '#1a1a2e',
                color: val > 0 ? '#fff' : '#333', borderRadius: '2px', border: '1px solid #222',
              }}>
                {val}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '4px 10px', fontSize: '14px', background: '#1a1a2e', color: '#e0e0e0',
  border: '1px solid #333', borderRadius: '4px', cursor: 'pointer',
  width: '36px', textAlign: 'center',
};

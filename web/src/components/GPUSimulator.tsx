import { useState, useEffect, useCallback, useRef } from 'react';
import { TinyGPUSim } from '../simulator/TinyGPUSim';
import { Instruction, SimulationState, PipelineStage } from '../compiler/types';

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

// RISC-V ABI register names
const RV_ABI = [
  'zero','ra','sp','gp','tp',
  't0','t1','t2','s0','s1',
  'a0','a1','a2','a3','a4','a5','a6','a7',
  's2','s3','s4','s5','s6','s7','s8',
  'tid','bid','bdm',  // x25=threadIdx, x26=blockIdx, x27=blockDim
  't3','t4','t5','t6',
];

export function GPUSimulator({
  instructions,
  initialMemory,
  numBlocks,
  threadsPerBlock,
  onCycleChange,
}: IMCSimulatorProps) {
  const [history, setHistory] = useState<SimulationState[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [selectedThread, setSelectedThread] = useState<number | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (instructions.length === 0) return;
    const sim = new TinyGPUSim(instructions, initialMemory, numBlocks, threadsPerBlock);
    const allStates = sim.runToEnd(5000);
    setHistory(allStates);
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
        setCurrentStep((prev) => {
          if (prev >= history.length - 1) { setIsPlaying(false); return prev; }
          return prev + 1;
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
    ? state.threads.find(t => t.threadId === selectedThread)
    : null;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Controls */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '6px', padding: '8px',
        background: '#111', borderRadius: '4px', flexShrink: 0, flexWrap: 'wrap',
      }}>
        <button onClick={reset} style={btnStyle} title="Reset">{'\u23EE'}</button>
        <button onClick={stepBackward} style={btnStyle} title="Step Back">{'\u23EA'}</button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={{ ...btnStyle, background: isPlaying ? '#e06c75' : '#2d5a3d', width: '48px' }}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '\u23F8' : '\u25B6'}
        </button>
        <button onClick={stepForward} style={btnStyle} title="Step Forward">{'\u23E9'}</button>
        <button onClick={jumpToEnd} style={btnStyle} title="Jump to End">{'\u23ED'}</button>
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

      {/* Block info */}
      <div style={{ fontSize: '11px', color: '#888', padding: '0 4px', flexShrink: 0, display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span>Block {state.currentBlock} / {state.totalBlocks}</span>
        {state.currentBlock >= state.totalBlocks && <span style={{ color: '#4ec9b0' }}>DONE</span>}
        {state.threads.some(t => t.divergent) && (
          <span style={{ color: '#e06c75', padding: '1px 6px', background: '#3a2020', borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>
            DIVERGENT
          </span>
        )}
        {state.threads.some(t => t.stage === PipelineStage.BARRIER) && (
          <span style={{ color: '#4ec9b0', padding: '1px 6px', background: '#1a3a3a', borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>
            FENCE/SYNC
          </span>
        )}
      </div>

      {/* Thread grid */}
      <div style={{ flex: 1, overflow: 'auto' }}>
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
              onClick={() => setSelectedThread(selectedThread === thread.threadId ? null : thread.threadId)}
            />
          ))}
        </div>

        {/* Selected thread register file */}
        {selectedThreadState && (
          <div style={{ margin: '8px 4px', padding: '8px', background: '#1a1a2e', border: '1px solid #4ec9b0', borderRadius: '6px' }}>
            <div style={{ fontSize: '11px', color: '#4ec9b0', fontWeight: 700, marginBottom: '6px' }}>
              RISC-V Core T{selectedThreadState.threadId} — Register File (x0–x31)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px', fontSize: '10px', fontFamily: 'monospace' }}>
              {selectedThreadState.registers.map((val, r) => {
                const isSpecial = r === 25 || r === 26 || r === 27;
                const isNonzero = val !== 0 && r !== 0;
                return (
                  <div key={r} style={{
                    padding: '3px 4px',
                    background: isSpecial ? '#2a2a4a' : isNonzero ? '#1a2a1a' : '#111',
                    borderRadius: '3px',
                    color: isSpecial ? '#c586c0' : isNonzero ? '#98c379' : '#444',
                  }}>
                    <span style={{ color: '#888' }}>{RV_ABI[r] ?? `x${r}`}</span>{' '}{val}
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

        {/* Memristor Crossbar Visualization */}
        <div style={{ margin: '8px 4px' }}>
          <div style={{ fontSize: '11px', color: '#4ec9b0', marginBottom: '4px', fontWeight: 600 }}>
            Memristor Crossbar (16×16 conductance grid)
          </div>
          <CrossbarGrid crossbar={state.crossbar} />
        </div>

        {/* Memory visualization */}
        <div style={{ marginTop: '8px', padding: '4px' }}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Global Memory (256 bytes)
          </div>
          <MemoryHeatmap memory={state.memory} />
        </div>

        {/* Shared (scratchpad) memory */}
        {state.sharedMemory.some(v => v !== 0) && (
          <div style={{ marginTop: '8px', padding: '4px' }}>
            <div style={{ fontSize: '11px', color: '#4ec9b0', marginBottom: '4px' }}>
              Scratchpad Memory (64 bytes)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1px' }}>
              {state.sharedMemory.slice(0, 32).map((val, i) => (
                <div key={i} style={{
                  width: '20px', height: '18px', display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontSize: '9px', fontFamily: 'monospace',
                  background: val > 0 ? `#4ec9b0${Math.min(Math.floor(val / 255 * 200) + 55, 255).toString(16).padStart(2, '0')}` : '#1a1a2e',
                  color: val > 0 ? '#fff' : '#333', borderRadius: '2px', border: '1px solid #222',
                }} title={`[S${i}] = ${val}`}>
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

function CrossbarGrid({ crossbar }: { crossbar: import('../compiler/types').CrossbarState }) {
  const size = 8; // Show 8×8 for display
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${size}, 1fr)`, gap: '1px', maxWidth: '160px' }}>
        {crossbar.conductances.slice(0, size).map((row, r) =>
          row.slice(0, size).map((g, c) => (
            <div
              key={`${r}-${c}`}
              style={{
                width: '18px', height: '18px',
                background: `rgba(78, 201, 176, ${g / 255 * 0.9 + 0.05})`,
                borderRadius: '1px',
                border: '1px solid #222',
              }}
              title={`G[${r}][${c}] = ${g}`}
            />
          ))
        )}
      </div>
      <div style={{ fontSize: '9px', color: '#555', marginTop: '2px' }}>
        conductance: low
        <span style={{ display: 'inline-block', width: '30px', height: '4px', background: 'linear-gradient(to right, #1a1a2e, #4ec9b0)', margin: '0 4px', verticalAlign: 'middle', borderRadius: '2px' }} />
        high
      </div>
      {crossbar.lastMVMResult.some(v => v !== 0) && (
        <div style={{ marginTop: '4px', fontSize: '10px', color: '#4ec9b0' }}>
          Last MVM output: [{crossbar.lastMVMResult.slice(0, 8).join(', ')}…]
        </div>
      )}
    </div>
  );
}

function ThreadCard({
  thread, selected, onClick,
}: {
  thread: import('../compiler/types').ThreadState;
  selected: boolean;
  onClick: () => void;
}) {
  const stageColor = STAGE_COLORS[thread.stage] || '#555';
  // Show first 8 allocatable registers (x5-x12)
  const allocRegs = thread.registers.slice(5, 13);

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
            <span style={{ background: '#3a2020', color: '#e06c75', padding: '1px 4px', borderRadius: '3px', fontSize: '8px', fontWeight: 700 }}>
              DIV
            </span>
          )}
          <span style={{ background: stageColor, color: '#fff', padding: '1px 6px', borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>
            {thread.stage}
          </span>
        </div>
      </div>
      <div style={{ color: '#888', marginBottom: '4px' }}>
        PC: <span style={{ color: '#b5cea8' }}>{thread.pc}</span>
        {thread.currentInstruction && (
          <span style={{ color: '#c586c0', marginLeft: '8px' }}>{thread.currentInstruction}</span>
        )}
      </div>
      {/* Compact register display (t0-t2, s0-s1) */}
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
              <div key={i} style={{
                width: '20px', height: '18px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '9px', fontFamily: 'monospace',
                background: val > 0 ? `${region.color}${Math.min(Math.floor(val / 255 * 200) + 55, 255).toString(16).padStart(2, '0')}` : '#1a1a2e',
                color: val > 0 ? '#fff' : '#333', borderRadius: '2px', border: '1px solid #222',
              }} title={`[${region.start + i}] = ${val}`}>
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
  border: '1px solid #333', borderRadius: '4px', cursor: 'pointer', width: '36px', textAlign: 'center',
};

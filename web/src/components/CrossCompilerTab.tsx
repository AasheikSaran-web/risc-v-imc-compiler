/**
 * CrossCompilerTab.tsx
 * C / Python → RISC-V Crossbar Assembly compiler tab.
 */

import { useState, useCallback, useRef } from 'react';
import { crossCompile, Lang, CompileStep, DEFAULT_C_SOURCE, DEFAULT_PY_SOURCE } from '../compiler/CrossCompiler';
import { CrossCompileResult } from '../compiler/CrossCompiler';

interface CrossCompilerTabProps {}

// ── Step color config ──────────────────────────────────────────────────────────
const STEP_COLORS: string[] = [
  '#61afef',  // 1 Tokenization  — blue
  '#c586c0',  // 2 AST           — purple
  '#e5c07b',  // 3 IR            — orange
  '#98c379',  // 4 Variable Anal — green
  '#4ec9b0',  // 5 Memory Map    — teal
  '#e06c75',  // 6 Assembly      — red
];

// ── Syntax highlighting for assembly ──────────────────────────────────────────
function highlightAsm(line: string): JSX.Element {
  // Comment
  if (/^\s*;/.test(line)) return <span style={{ color: '#98c379' }}>{line}</span>;
  // Label
  if (/^[a-zA-Z_]\w*:/.test(line.trimStart())) return <span style={{ color: '#e5c07b' }}>{line}</span>;
  // Crossbar mnemonics
  if (/\b(cvld|cvst|xadd|xsub|xmul|mvm|cset)\b/i.test(line)) return <span style={{ color: '#4ec9b0' }}>{line}</span>;
  // Directives
  if (/^\s*\./.test(line)) return <span style={{ color: '#d19a66' }}>{line}</span>;
  return <span style={{ color: '#e0e0e0' }}>{line}</span>;
}

// ── Collapsible step ──────────────────────────────────────────────────────────
interface StepSectionProps {
  step: CompileStep;
  color: string;
  defaultOpen: boolean;
  index: number;
}

function StepSection({ step, color, defaultOpen, index }: StepSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div style={{ borderBottom: '1px solid #2a2a3e', marginBottom: '2px' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', textAlign: 'left', padding: '6px 10px',
          background: open ? `${color}18` : '#161628',
          border: 'none', borderLeft: `3px solid ${color}`,
          color: color, cursor: 'pointer', fontFamily: 'monospace',
          fontSize: '11px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px',
        }}
      >
        <span style={{ fontSize: '9px' }}>{open ? '▼' : '▶'}</span>
        {step.title}
      </button>
      {open && (
        <pre style={{
          margin: 0, padding: '8px 12px',
          background: '#0d0d1a', color: '#cdd6f4',
          fontSize: '10.5px', fontFamily: 'monospace',
          overflowX: 'auto', whiteSpace: 'pre-wrap',
          wordBreak: 'break-word', maxHeight: '220px',
          overflowY: 'auto', borderLeft: `3px solid ${color}40`,
        }}>
          {step.content}
        </pre>
      )}
    </div>
  );
}

// ── Main tab component ─────────────────────────────────────────────────────────
export function CrossCompilerTab(_props: CrossCompilerTabProps) {
  const [lang, setLang]       = useState<Lang>('c');
  const [source, setSource]   = useState(DEFAULT_C_SOURCE);
  const [result, setResult]   = useState<CrossCompileResult | null>(null);
  const [compiling, setCompiling] = useState(false);

  const handleLangChange = useCallback((newLang: Lang) => {
    setLang(newLang);
    setSource(newLang === 'c' ? DEFAULT_C_SOURCE : DEFAULT_PY_SOURCE);
    setResult(null);
  }, []);

  const handleCompile = useCallback(() => {
    setCompiling(true);
    // Run in next tick so the UI can update first
    setTimeout(() => {
      try {
        const r = crossCompile(source, lang);
        setResult(r);
      } catch {
        setResult(null);
      }
      setCompiling(false);
    }, 0);
  }, [source, lang]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#111', color: '#e0e0e0' }}>

      {/* ── Header bar ─────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', padding: '6px 14px',
        background: '#111', borderBottom: '1px solid #222', gap: '10px', flexShrink: 0,
      }}>
        <span style={{ fontFamily: 'monospace', fontSize: '12px', color: '#4ec9b0', fontWeight: 700 }}>
          C/Python → Crossbar Assembly
        </span>
        <span style={{ color: '#444', fontSize: '12px' }}>|</span>

        {/* Language toggle */}
        {(['c', 'python'] as Lang[]).map(l => (
          <button
            key={l}
            onClick={() => handleLangChange(l)}
            style={{
              padding: '3px 12px', fontSize: '11px', fontFamily: 'monospace',
              background: lang === l ? '#2d5a3d' : 'transparent',
              color:      lang === l ? '#7fff7f' : '#888',
              border:     lang === l ? '1px solid #4a8a5a' : '1px solid #333',
              borderRadius: '4px', cursor: 'pointer',
            }}
          >
            {l === 'c' ? 'C' : 'Python'}
          </button>
        ))}

        <div style={{ flex: 1 }} />
        <span style={{ fontSize: '10px', color: '#555', fontFamily: 'monospace' }}>
          custom-0: MVM/CSET · custom-1: XADD/XSUB/XMUL · custom-2: CVLD/CVST
        </span>
      </div>

      {/* ── Main area ──────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* ── Left: Source editor ──────────────────────────────────────────── */}
        <div style={{
          width: '30%', minWidth: '220px', display: 'flex', flexDirection: 'column',
          borderRight: '1px solid #222',
        }}>
          {/* Panel header */}
          <div style={{
            padding: '5px 10px', background: '#111', fontSize: '10px', color: '#888',
            borderBottom: '1px solid #222', display: 'flex', alignItems: 'center', gap: '5px', flexShrink: 0,
          }}>
            <span style={{ color: '#e06c75' }}>●</span>
            Source · {lang === 'c' ? '.c' : '.py'}
          </div>

          <textarea
            value={source}
            onChange={e => setSource(e.target.value)}
            spellCheck={false}
            style={{
              flex: 1, background: '#1e1e1e', color: '#d4d4d4',
              fontFamily: '"Fira Code", "Cascadia Code", "JetBrains Mono", monospace',
              fontSize: '11.5px', lineHeight: '1.6',
              border: 'none', outline: 'none', resize: 'none',
              padding: '10px 12px', tabSize: 4,
            }}
          />

          {/* Compile button */}
          <div style={{ padding: '8px 10px', background: '#111', borderTop: '1px solid #222', flexShrink: 0 }}>
            <button
              onClick={handleCompile}
              disabled={compiling}
              style={{
                width: '100%', padding: '7px 0',
                background: compiling ? '#1e3a2a' : '#2d5a3d',
                color: compiling ? '#5a9a6a' : '#7fff7f',
                border: '1px solid #4a8a5a', borderRadius: '4px',
                cursor: compiling ? 'wait' : 'pointer',
                fontFamily: 'monospace', fontSize: '12px', fontWeight: 700,
              }}
            >
              {compiling ? 'Compiling...' : 'Compile →'}
            </button>
          </div>
        </div>

        {/* ── Center: Compilation steps ─────────────────────────────────────── */}
        <div style={{
          width: '35%', display: 'flex', flexDirection: 'column',
          borderRight: '1px solid #222', overflow: 'hidden',
        }}>
          <div style={{
            padding: '5px 10px', background: '#111', fontSize: '10px', color: '#888',
            borderBottom: '1px solid #222', flexShrink: 0,
          }}>
            <span style={{ color: '#c586c0' }}>●</span> Compilation Steps
          </div>

          <div style={{ flex: 1, overflowY: 'auto', background: '#0f0f1e' }}>
            {!result && (
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: '#444', fontFamily: 'monospace', fontSize: '12px',
                flexDirection: 'column', gap: '8px',
              }}>
                <span style={{ fontSize: '28px' }}>⌨</span>
                <span>Write some {lang === 'c' ? 'C' : 'Python'} code and click Compile →</span>
              </div>
            )}

            {result && result.steps.map((step, i) => (
              <StepSection
                key={i}
                step={step}
                color={STEP_COLORS[i] ?? '#888'}
                defaultOpen={true}
                index={i}
              />
            ))}
          </div>
        </div>

        {/* ── Right: Assembly output ────────────────────────────────────────── */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{
            padding: '5px 10px', background: '#111', fontSize: '10px', color: '#888',
            borderBottom: '1px solid #222', flexShrink: 0, display: 'flex', alignItems: 'center', gap: '6px',
          }}>
            <span style={{ color: '#98c379' }}>●</span> Assembly Output
            {result && (
              <>
                <span style={{ marginLeft: 'auto' }} />
                {/* Utilization badge */}
                <span style={{
                  padding: '2px 8px', background: '#0d2d2d',
                  border: '1px solid #4ec9b0', borderRadius: '10px',
                  color: '#4ec9b0', fontSize: '10px', fontFamily: 'monospace',
                }}>
                  Crossbar: {result.mapping.utilizationFactor}% ({result.mapping.usedCells}/2880)
                </span>
                <span style={{
                  padding: '2px 8px', background: '#1a2d1a',
                  border: '1px solid #98c379', borderRadius: '10px',
                  color: '#98c379', fontSize: '10px', fontFamily: 'monospace',
                }}>
                  Regs: {result.mapping.registerCount}/20
                </span>
              </>
            )}
          </div>

          {/* Errors */}
          {result && result.errors.length > 0 && (
            <div style={{ padding: '6px 12px', background: '#2d1111', borderBottom: '1px solid #5a2222', flexShrink: 0 }}>
              {result.errors.map((err, i) => (
                <div key={i} style={{ color: '#e06c75', fontFamily: 'monospace', fontSize: '10px' }}>
                  ⚠ {err}
                </div>
              ))}
            </div>
          )}

          {/* Assembly text */}
          <div style={{ flex: 1, overflowY: 'auto', background: '#1a1a2e' }}>
            {!result && (
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                height: '100%', color: '#444', fontFamily: 'monospace', fontSize: '12px',
                flexDirection: 'column', gap: '8px',
              }}>
                <span style={{ fontSize: '28px' }}>🔩</span>
                <span>Assembly will appear here after compilation</span>
              </div>
            )}
            {result && (
              <pre style={{
                margin: 0, padding: '10px 14px',
                fontFamily: '"Fira Code", "Cascadia Code", "JetBrains Mono", monospace',
                fontSize: '10.5px', lineHeight: '1.65', background: 'transparent',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              }}>
                {result.assembly.split('\n').map((line, i) => (
                  <div key={i}>{highlightAsm(line)}{'\n'}</div>
                ))}
              </pre>
            )}
          </div>

          {/* Stats footer */}
          {result && (
            <div style={{
              padding: '5px 12px', background: '#111', borderTop: '1px solid #222',
              display: 'flex', gap: '16px', flexShrink: 0, flexWrap: 'wrap',
            }}>
              <span style={{ color: '#4ec9b0', fontFamily: 'monospace', fontSize: '10px' }}>
                Crossbar Utilization: {result.mapping.utilizationFactor}% ({result.mapping.usedCells} / 2880 cells)
              </span>
              <span style={{ color: '#98c379', fontFamily: 'monospace', fontSize: '10px' }}>
                Registers: {result.mapping.registerCount} / 20 GPRs
              </span>
              <span style={{ color: '#e5c07b', fontFamily: 'monospace', fontSize: '10px' }}>
                Crossbar vars: {result.mapping.crossbarVarCount}
              </span>
              <span style={{ color: '#c586c0', fontFamily: 'monospace', fontSize: '10px' }}>
                Total cells: 45 cols × 64 rows = 2880
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * MemoryMapper.ts
 * Crossbar memory mapping algorithm for the 64×64 memristor crossbar.
 *
 * Physical layout (col indices):
 *   Col  0      — register-file backing (reserved)
 *   Cols 1–16   — MVM weight matrix (CSET only, reserved)
 *   Cols 17–61  — variable data region (CVLD/CVST safe zone) → 45 cols × 64 rows = 2880 cells
 *   Col  62     — XA arithmetic scratch (reserved)
 *   Col  63     — reserved
 */

export interface VariableInfo {
  name: string;
  isArray: boolean;
  size: number;       // 1 for scalar, N for array
  defLine: number;
  lastUseLine: number;
  accessCount: number;
}

export interface CellAlloc { row: number; col: number; }

export interface VarAllocation {
  variable: string;
  storage: 'register' | 'crossbar';
  register?: string;  // e.g. 't0'
  cells?: CellAlloc[];
  size: number;
}

export interface MappingResult {
  allocations: VarAllocation[];
  utilizationFactor: number;  // 0.0–100.0
  usedCells: number;
  totalCells: number;         // always 2880 (45 cols × 64 rows)
  registerCount: number;
  crossbarVarCount: number;
  heatmap: number[][];        // [row 0..63][col 0..44] — 1 if occupied
}

// ── Register file: 20 GPRs ──────────────────────────────────────────────────
const REGISTERS = [
  't0','t1','t2','t3','t4','t5','t6',       // 7 temporaries
  's0','s1','s2','s3','s4','s5','s6','s7',  // 8 saved
  'a0','a1','a2','a3','a4',                  // 5 argument/return
];

// ── Crossbar data region constants ──────────────────────────────────────────
const CV_COL_START  = 17;
const CV_COL_END    = 61;
const CV_COLS       = CV_COL_END - CV_COL_START + 1;  // 45
const CV_ROWS       = 64;
const TOTAL_CELLS   = CV_COLS * CV_ROWS;               // 2880

/**
 * Check whether two lifetime intervals overlap.
 * Interval A: [defA, lastUseA]   Interval B: [defB, lastUseB]
 * They DON'T overlap if A ends before B starts, or B ends before A starts.
 */
function lifetimesOverlap(a: VariableInfo, b: VariableInfo): boolean {
  return !(a.lastUseLine < b.defLine || b.lastUseLine < a.defLine);
}

export function mapVariables(variables: VariableInfo[]): MappingResult {
  const allocations: VarAllocation[] = [];
  const heatmap: number[][] = Array.from({ length: CV_ROWS }, () => new Array(CV_COLS).fill(0));

  // ── 1. Partition into register candidates and crossbar candidates ──────────
  const scalars = variables.filter(v => !v.isArray).sort((a, b) => b.accessCount - a.accessCount);
  const arrays  = variables.filter(v => v.isArray);

  // ── 2. Assign hot scalars to registers (first-fit by accessCount desc) ─────
  const regPool = [...REGISTERS];
  const regVars: VariableInfo[] = [];
  const spilledScalars: VariableInfo[] = [];

  for (const v of scalars) {
    if (regPool.length > 0) {
      const reg = regPool.shift()!;
      allocations.push({ variable: v.name, storage: 'register', register: reg, size: v.size });
      regVars.push(v);
    } else {
      spilledScalars.push(v);
    }
  }

  // ── 3. Build list of crossbar variables (arrays + spilled scalars) ─────────
  // Sort by accessCount descending for spatial locality (hot vars → lower row indices).
  const crossbarVars: VariableInfo[] = [
    ...arrays,
    ...spilledScalars,
  ].sort((a, b) => b.accessCount - a.accessCount);

  // ── 4. Lifetime-aware interval coloring ────────────────────────────────────
  // We track which cells are occupied and by which variable's lifetime.
  // A cell can be reused if the previous occupant's lastUseLine < current var's defLine.
  //
  // Data structure: for each crossbar cell [row][col] track the variable currently
  // "owning" it (by index into crossbarVars or -1 if free).
  // We iterate column-major (col 17→61, row 0→63).

  // cellOwner[row][relCol] = index into crossbarVars, or -1
  const cellOwner: number[][] = Array.from({ length: CV_ROWS }, () => new Array(CV_COLS).fill(-1));

  let usedCells = 0;

  for (let vi = 0; vi < crossbarVars.length; vi++) {
    const v = crossbarVars[vi];
    const needed = v.size;
    const cells: CellAlloc[] = [];

    // Try to find `needed` consecutive or individual cells via lifetime reuse.
    // Scan column-major.
    let found = 0;
    outer:
    for (let relCol = 0; relCol < CV_COLS && found < needed; relCol++) {
      for (let row = 0; row < CV_ROWS && found < needed; row++) {
        const ownerIdx = cellOwner[row][relCol];
        if (ownerIdx === -1) {
          // Free cell — allocate.
          cellOwner[row][relCol] = vi;
          heatmap[row][relCol] = 1;
          cells.push({ row, col: relCol + CV_COL_START });
          found++;
          usedCells++;
        } else {
          // Cell occupied — check if lifetimes don't overlap (reuse eligible).
          const owner = crossbarVars[ownerIdx];
          if (!lifetimesOverlap(owner, v)) {
            // Reuse: update owner to current var (later lifetime takes over).
            cellOwner[row][relCol] = vi;
            // heatmap already 1; usedCells already counted.
            cells.push({ row, col: relCol + CV_COL_START });
            found++;
          }
        }
      }
    }

    // If we couldn't place all cells (crossbar full), place what we can.
    if (found < needed) {
      // Still emit the allocation with however many cells we found.
    }

    allocations.push({
      variable: v.name,
      storage: 'crossbar',
      cells,
      size: v.size,
    });
  }

  // ── 5. Count lifetime-reuse pairs ─────────────────────────────────────────
  // (informational only; not returned directly but used in CrossCompiler display)

  const utilizationFactor = Math.round((usedCells / TOTAL_CELLS) * 1000) / 10;

  return {
    allocations,
    utilizationFactor,
    usedCells,
    totalCells: TOTAL_CELLS,
    registerCount: regVars.length,
    crossbarVarCount: crossbarVars.length,
    heatmap,
  };
}

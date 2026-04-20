/** Represents a single 32-bit RISC-V instruction */
export interface Instruction {
  addr: number;
  hex: string;   // 8-char hex (32-bit)
  asm: string;
  bits: string;  // 32 bits, space-separated in groups of 8
}

/** A single compilation stage showing IR at that point */
export interface CompilationStage {
  name: string;
  ir: string;
}

/** Thread divergence info */
export interface DivergenceInfo {
  instructionAddr: number;
  type: 'branch' | 'converge';
  branchTaken: boolean[];
  description: string;
}

/** Memory access pattern analysis */
export interface CoalescingInfo {
  instructionAddr: number;
  accessPattern: 'coalesced' | 'strided' | 'scattered';
  addresses: number[];
  transactionsNeeded: number;
  description: string;
}

/** Performance metrics with IMC-specific additions */
export interface PerformanceMetrics {
  totalInstructions: number;
  registersUsed: number;
  sharedMemoryBytes: number;
  branchInstructions: number;
  memoryInstructions: number;
  computeInstructions: number;
  imcOperations: number;         // MVM crossbar instructions
  crossbarWriteOps: number;      // CSET instructions
  crossbarArithOps: number;      // XADD + XSUB + XMUL instructions (custom-1)
  cvLoadOps: number;             // CVLD crossbar variable load (custom-2)
  cvStoreOps: number;            // CVST crossbar variable store (custom-2)
  barrierCount: number;
  estimatedCycles: number;
  crossbarCyclesSaved: number;   // Cycles saved vs scalar multiply-accumulate
  computeToMemoryRatio: number;
  optimizationSummary: string;
}

/** Full analysis results */
export interface AnalysisResult {
  divergence: DivergenceInfo[];
  coalescing: CoalescingInfo[];
  metrics: PerformanceMetrics;
}

/** Full compilation trace */
export interface CompilationTrace {
  source: string;
  stages: CompilationStage[];
  binary: {
    instructions: Instruction[];
  };
  analysis?: AnalysisResult;
}

// Pre-compute G_STEP to avoid self-reference in object literal
const _G_STEP_US = (5900.0 - 0.2) / (16520 - 1);  // ≈ 0.35715 µS/level

/**
 * Physical memristor device constants from:
 * "Linear symmetric self-selecting 14-bit kinetic molecular memristors"
 * Nature 633, Sep 2024 — [RuIIL](BF₄)₂ on a 64×64 crossbar
 */
export const MEMRISTOR_PHYSICS = {
  G_MIN_US:      0.2,          // µS  — minimum conductance (200 nS)
  G_MAX_US:      5900.0,       // µS  — maximum conductance (5.9 mS)
  LEVELS:        16520,        // analog conductance levels (≈14-bit)
  G_STEP_US:     _G_STEP_US,  // µS per level  ≈ 0.35715
  POT_V_MV:      900,          // mV  — potentiation pulse voltage
  POT_NS:        80,           // ns  — potentiation pulse width
  DEP_V_MV:      750,          // mV  — depression pulse voltage (absolute)
  DEP_NS:        65,           // ns  — depression pulse width
  VTH_MV:        830,          // mV  — write threshold voltage
  READ_V_MV:     500,          // mV  — non-disturbing read voltage
  HALF_SEL_MV:   450,          // mV  — half-select voltage (below VTH)
  RMSE_NS:       42,           // nS  — write accuracy RMSE target → achieved
  SNR_DB:        73,           // dB  — signal-to-noise ratio (73–79 dB range)
  CROSSBAR_SIZE: 64,           // 64×64 molecular memristor array
  MATERIAL:      'Memristor-Based Crossbar',
  PAPER_REF:     'Nature 633, Sep 2024',
} as const;

/**
 * A single memristor write event: a potentiation or depression pulse sequence
 * that programs one crossbar cell to a new conductance level.
 */
export interface MemristorWriteEvent {
  cycle: number;
  row: number;
  col: number;
  prevG_us: number;             // conductance before write (µS)
  newG_us: number;              // conductance after write (µS)
  pulseType: 'pot' | 'dep' | 'hold';  // potentiation / depression / no change
  pulseCount: number;           // estimated pulses applied (ΔG / G_STEP)
  registerName?: string;        // ABI name if col=0 (register-file cell)
}

/** RISC-V base opcodes (7-bit) */
export enum RVOpcode {
  OP       = 0x33,  // R-type: ADD, SUB, MUL, DIV
  OP_IMM   = 0x13,  // I-type: ADDI
  LOAD     = 0x03,  // LW (global memory)
  STORE    = 0x23,  // SW (global memory)
  BRANCH   = 0x63,  // BEQ, BNE, BLT, BGE
  JAL      = 0x6F,  // Unconditional jump
  LUI      = 0x37,  // Load upper immediate
  CUSTOM0  = 0x0B,  // Custom-0: MVM / SLDR / SSTR / CSET (memristor crossbar)
  CUSTOM1  = 0x2B,  // Custom-1: XADD / XSUB / XMUL  (crossbar integer arithmetic)
  CUSTOM2  = 0x5B,  // Custom-2: CVLD / CVST           (crossbar variable load/store)
  MISC_MEM = 0x0F,  // FENCE (thread barrier)
  SYSTEM   = 0x73,  // ECALL (thread completion)
}

/** Human-readable opcode labels (for simulator display) */
export const OPCODE_NAMES: Record<number, string> = {
  [RVOpcode.OP]:       'R-type',
  [RVOpcode.OP_IMM]:   'ADDI',
  [RVOpcode.LOAD]:     'LW',
  [RVOpcode.STORE]:    'SW',
  [RVOpcode.BRANCH]:   'BRANCH',
  [RVOpcode.JAL]:      'JAL',
  [RVOpcode.LUI]:      'LUI',
  [RVOpcode.CUSTOM0]:  'IMC',
  [RVOpcode.CUSTOM1]:  'XA',
  [RVOpcode.CUSTOM2]:  'CVMEM',
  [RVOpcode.MISC_MEM]: 'FENCE',
  [RVOpcode.SYSTEM]:   'ECALL',
};

/** RISC-V 5-stage pipeline (simplified) */
export enum PipelineStage {
  FETCH     = 'FETCH',
  DECODE    = 'DECODE',
  EXECUTE   = 'EXECUTE',
  MEMORY    = 'MEMORY',
  WRITEBACK = 'WRITEBACK',
  BARRIER   = 'BARRIER',
  DONE      = 'DONE',
}

/** Per-thread/core RISC-V execution state */
export interface ThreadState {
  threadId: number;
  blockId: number;
  pc: number;
  registers: number[];       // x0–x31 (32-bit general purpose)
  stage: PipelineStage;
  done: boolean;
  currentInstruction: string;
  divergent?: boolean;
}

/**
 * 64×64 memristor crossbar state (all conductances in µS).
 *
 * Physical layout:
 *   Column 0         — register-file backing: row r stores the conductance
 *                      encoding of RISC-V register xr (r = 0..31).
 *   Columns 1–63     — weight matrix for MVM operations.
 *
 * The VMM formula (Kirchhoff's current law):
 *   I_Q = Σ_P  V_P · G_{P,Q}     (analog, single-cycle)
 */
export interface CrossbarState {
  conductances: number[][];          // 64×64 grid in µS
  lastMVMResult: number[];           // 16-element output of most recent MVM
  writeEvents: MemristorWriteEvent[]; // recent write history (capped at 20)
}

/**
 * Crossbar scratch region for custom-1 integer arithmetic.
 * Column 62 is dedicated to XADD/XSUB/XMUL operand and result cells.
 *
 *  XADD: rows 32(A), 33(B), 34(R)  — conductance superposition  G_R = G_A + G_B − G_MIN
 *  XSUB: rows 35(A), 36(B), 37(R)  — differential pair           G_R = G_A − G_B + G_MIN
 *  XMUL: rows 38(V), 39(G), 40(R)  — Ohm's law                   I   = V(A) · G(B)  → R
 */
export const XA_SCRATCH = {
  COL:        62,
  XADD_ROW_A: 32, XADD_ROW_B: 33, XADD_ROW_R: 34,
  XSUB_ROW_A: 35, XSUB_ROW_B: 36, XSUB_ROW_R: 37,
  XMUL_ROW_V: 38, XMUL_ROW_G: 39, XMUL_ROW_R: 40,
} as const;

/**
 * Crossbar variable data region for custom-2 CVLD/CVST.
 *
 * Memory mapping (safe zones only — no overlap with reserved regions):
 *   Col  0         — register-file backing (reserved)
 *   Cols 1–16      — MVM weight matrix     (reserved)
 *   Cols 17–61     — variable data region  ← CVLD/CVST safe zone
 *   Col  62        — XA arithmetic scratch (reserved)
 *   Col  63        — reserved
 *
 * Read path:  V_read ≤ 500 mV (below V_th = 830 mV) — non-disturbing.
 * Write path: pot/dep pulse sequence; bounds-enforced in hardware sim.
 */
export const CV_DATA = {
  COL_START: 17,   // first safe column for variable storage
  COL_END:   61,   // last safe column  (inclusive)
  ROW_COUNT: 64,   // all 64 rows available within the data region
} as const;

/** Full IMC simulation state at one cycle */
export interface SimulationState {
  cycle: number;
  threads: ThreadState[];
  memory: number[];            // 256-byte global data memory
  sharedMemory: number[];      // 64-byte shared scratchpad per block
  crossbar: CrossbarState;     // 64×64 memristor crossbar
  currentBlock: number;
  totalBlocks: number;
}

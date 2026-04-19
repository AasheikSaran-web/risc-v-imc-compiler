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

/** RISC-V base opcodes (7-bit) */
export enum RVOpcode {
  OP       = 0x33,  // R-type: ADD, SUB, MUL, DIV
  OP_IMM   = 0x13,  // I-type: ADDI
  LOAD     = 0x03,  // LW (global memory)
  STORE    = 0x23,  // SW (global memory)
  BRANCH   = 0x63,  // BEQ, BNE, BLT, BGE
  JAL      = 0x6F,  // Unconditional jump
  LUI      = 0x37,  // Load upper immediate
  CUSTOM0  = 0x0B,  // Custom: MVM / SLDR / SSTR (memristor crossbar)
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
  registers: number[];       // x0-x31 (32-bit general purpose)
  stage: PipelineStage;
  done: boolean;
  currentInstruction: string;
  divergent?: boolean;
}

/** Memristor crossbar state (16×16 conductance grid) */
export interface CrossbarState {
  conductances: number[][];    // 16×16 grid (0-255 = conductance level)
  lastMVMResult: number[];     // Output of the most recent MVM operation
}

/** Full IMC simulation state at one cycle */
export interface SimulationState {
  cycle: number;
  threads: ThreadState[];
  memory: number[];            // 256-byte global data memory
  sharedMemory: number[];      // 64-byte shared scratchpad per block
  crossbar: CrossbarState;     // Memristor crossbar array
  currentBlock: number;
  totalBlocks: number;
}

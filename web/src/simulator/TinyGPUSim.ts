// RISC-V + Memristor Crossbar In-Memory Compute simulator.
//
// Physical model: "Linear symmetric self-selecting 14-bit kinetic molecular
// memristors", Nature 633, Sep 2024 — [RuIIL](BF₄)₂, 64×64 crossbar.
//
// Register-file backing:
//   Every RISC-V register write internally programs crossbar cell (reg, 0)
//   via potentiation / depression pulse sequences (900mV / -750mV, 80ns/65ns).
//   On read the cell is probed at ≤500mV (non-disturbing).
//
// Custom-0 (opcode=0x0B) instruction set:
//   funct3=0x0  MVM  rd, rs1       — Analog matrix-vector multiply
//   funct3=0x1  SLDR rd, rs1       — Scratchpad load
//   funct3=0x2  SSTR rs1, rs2      — Scratchpad store
//   funct3=0x3  CSET rd, rs1, rs2  — Program crossbar[rs1][rs2] = reg[rd]
//
// Custom-1 (opcode=0x2B) instruction set — crossbar integer arithmetic:
//   funct3=0x0  XADD rd, rs1, rs2  — Add:      G_R = G(A)+G(B)−G_MIN → A+B
//   funct3=0x1  XSUB rd, rs1, rs2  — Subtract: G_R = G(A)−G(B)+G_MIN → A−B
//   funct3=0x2  XMUL rd, rs1, rs2  — Multiply: I   = V(A)·G(B)        → A×B

import {
  Instruction,
  RVOpcode,
  OPCODE_NAMES,
  PipelineStage,
  ThreadState,
  SimulationState,
  CrossbarState,
  MemristorWriteEvent,
  MEMRISTOR_PHYSICS,
  XA_SCRATCH,
} from '../compiler/types';

// ── RISC-V ABI register names ─────────────────────────────────────────────
const RV_ABI = [
  'zero','ra','sp','gp','tp',
  't0','t1','t2','s0','s1',
  'a0','a1','a2','a3','a4','a5','a6','a7',
  's2','s3','s4','s5','s6','s7','s8',
  'tid','bid','bdm',
  't3','t4','t5','t6',
];

// ── Memristor conductance mapping ─────────────────────────────────────────
// Maps an 8-bit value (0–255) ↔ conductance in µS [G_MIN, G_MAX].
// Uses the lowest 8 bits of the 32-bit register value (practical data range).

const { G_MIN_US, G_MAX_US, G_STEP_US, LEVELS, READ_V_MV } = MEMRISTOR_PHYSICS;

export function valueToConductance(value: number): number {
  const v8 = (value >>> 0) & 0xFF;
  const level = Math.round(v8 / 255 * (LEVELS - 1));
  return G_MIN_US + level * G_STEP_US;
}

export function conductanceToValue(G: number): number {
  const level = Math.max(0, Math.min(LEVELS - 1,
    Math.round((G - G_MIN_US) / G_STEP_US)
  ));
  return Math.round(level / (LEVELS - 1) * 255);
}

// ── Sign-extend helper ────────────────────────────────────────────────────
function signExt(value: number, bits: number): number {
  const mask = 1 << (bits - 1);
  return (value & mask) ? (value | (-1 << bits)) : (value & (mask - 1 + mask));
}

/** Decode a 32-bit RISC-V instruction into its field components */
function decode32(instr: number) {
  const opcode  = instr & 0x7F;
  const rd      = (instr >> 7) & 0x1F;
  const funct3  = (instr >> 12) & 0x7;
  const rs1     = (instr >> 15) & 0x1F;
  const rs2     = (instr >> 20) & 0x1F;
  const funct7  = (instr >> 25) & 0x7F;

  const immI = signExt((instr >>> 20), 12);
  const immS = signExt(((instr >>> 25) << 5) | ((instr >> 7) & 0x1F), 12);
  const immB = signExt(
    (((instr >>> 31) & 1) << 12) |
    (((instr >> 7)  & 1) << 11) |
    (((instr >>> 25) & 0x3F) << 5) |
    (((instr >> 8)  & 0xF) << 1),
    13
  );
  const immJ = signExt(
    (((instr >>> 31) & 1) << 20) |
    (((instr >> 12) & 0xFF) << 12) |
    (((instr >>> 20) & 1) << 11) |
    (((instr >>> 21) & 0x3FF) << 1),
    21
  );

  return { opcode, rd, funct3, rs1, rs2, funct7, immI, immS, immB, immJ };
}

// ── Simulator ─────────────────────────────────────────────────────────────

export class TinyGPUSim {
  private program: number[] = [];
  private memory: number[] = [];
  private sharedMemory: number[] = [];
  private crossbar: CrossbarState;
  private threads: ThreadState[] = [];
  private numBlocks: number;
  private threadsPerBlock: number;
  private currentBlock = 0;
  private cycle = 0;

  constructor(
    instructions: Instruction[],
    initialMemory: number[],
    numBlocks: number,
    threadsPerBlock: number
  ) {
    this.numBlocks = numBlocks;
    this.threadsPerBlock = threadsPerBlock;

    this.program = instructions.map((i) => parseInt(i.hex.replace('0x', ''), 16) >>> 0);

    this.memory = new Array(256).fill(0);
    for (let i = 0; i < initialMemory.length && i < 256; i++) {
      this.memory[i] = initialMemory[i] & 0xFF;
    }

    this.sharedMemory = new Array(64).fill(0);

    // 64×64 crossbar — all cells start at G_MIN (inactive / erased state)
    const conductances: number[][] = Array.from(
      { length: 64 },
      () => Array.from({ length: 64 }, () => G_MIN_US)
    );

    this.crossbar = {
      conductances,
      lastMVMResult: new Array(16).fill(0),
      writeEvents: [],
    };

    this.initBlock(0);
  }

  // ── Block initialization ───────────────────────────────────────────────

  private initBlock(blockId: number) {
    this.currentBlock = blockId;
    this.threads = [];
    this.sharedMemory = new Array(64).fill(0);

    for (let t = 0; t < this.threadsPerBlock; t++) {
      const regs = new Array(32).fill(0);
      regs[25] = t;                        // x25 = threadIdx
      regs[26] = blockId;                  // x26 = blockIdx
      regs[27] = this.threadsPerBlock;     // x27 = blockDim

      this.threads.push({
        threadId: t,
        blockId,
        pc: 0,
        registers: regs,
        stage: PipelineStage.FETCH,
        done: false,
        currentInstruction: '',
        divergent: false,
      });

      // Mirror the preset special registers into crossbar col 0 (register-file backing)
      this.writeMemristorCell(25, 0, t,                    `tid=${t} (block ${blockId})`);
      this.writeMemristorCell(26, 0, blockId,              `bid=${blockId}`);
      this.writeMemristorCell(27, 0, this.threadsPerBlock, `bdm=${this.threadsPerBlock}`);
    }
  }

  // ── Memristor write helpers ────────────────────────────────────────────

  /**
   * Programs crossbar cell (row, col) to the conductance encoding of `value`.
   * Computes the potentiation/depression pulse count needed and logs a write event.
   */
  private writeMemristorCell(
    row: number, col: number, value: number, label?: string
  ): void {
    const prevG = this.crossbar.conductances[row][col];
    const newG  = valueToConductance(value);
    this.crossbar.conductances[row][col] = newG;

    const dG = newG - prevG;
    const pulseCount = Math.max(1, Math.round(Math.abs(dG) / G_STEP_US));
    const pulseType: MemristorWriteEvent['pulseType'] =
      dG > G_STEP_US ? 'pot' : dG < -G_STEP_US ? 'dep' : 'hold';

    const event: MemristorWriteEvent = {
      cycle: this.cycle,
      row, col,
      prevG_us: prevG,
      newG_us:  newG,
      pulseType,
      pulseCount,
      registerName: col === 0
        ? (label ?? RV_ABI[row] ?? `x${row}`)
        : undefined,
    };

    this.crossbar.writeEvents.push(event);
    if (this.crossbar.writeEvents.length > 20) this.crossbar.writeEvents.shift();
  }

  /**
   * Write to a RISC-V register AND mirror the value into crossbar col 0.
   * x0 is hardwired to 0 and never written.
   */
  private writeReg(thread: ThreadState, reg: number, value: number): void {
    if (reg === 0) return;
    thread.registers[reg] = value;
    // Crossbar register-file backing: cell (reg, 0) stores the conductance
    // encoding of this register's value via pot/dep pulse sequences.
    this.writeMemristorCell(reg, 0, value);
  }

  // ── State snapshot ─────────────────────────────────────────────────────

  getState(): SimulationState {
    return {
      cycle: this.cycle,
      threads: this.threads.map((t) => ({ ...t, registers: [...t.registers] })),
      memory: [...this.memory],
      sharedMemory: [...this.sharedMemory],
      crossbar: {
        conductances: this.crossbar.conductances.map((row) => [...row]),
        lastMVMResult: [...this.crossbar.lastMVMResult],
        writeEvents: this.crossbar.writeEvents.map((e) => ({ ...e })),
      },
      currentBlock: this.currentBlock,
      totalBlocks: this.numBlocks,
    };
  }

  isDone(): boolean {
    return this.currentBlock >= this.numBlocks;
  }

  // ── Simulation step ────────────────────────────────────────────────────

  step(): SimulationState {
    if (this.isDone()) return this.getState();

    if (this.threads.every((t) => t.done)) {
      this.currentBlock++;
      if (this.currentBlock < this.numBlocks) this.initBlock(this.currentBlock);
      this.cycle++;
      return this.getState();
    }

    // Barrier: hold all until everyone arrives
    const atBarrier = this.threads.filter((t) => !t.done && t.stage === PipelineStage.BARRIER);
    if (atBarrier.length > 0) {
      const active = this.threads.filter((t) => !t.done);
      if (atBarrier.length === active.length) {
        for (const t of atBarrier) { t.stage = PipelineStage.FETCH; t.pc++; }
      } else {
        for (const t of this.threads) {
          if (!t.done && t.stage !== PipelineStage.BARRIER) this.executeThread(t);
        }
      }
      this.cycle++;
      return this.getState();
    }

    // Divergence tracking
    const active = this.threads.filter((t) => !t.done);
    const pcCounts: Record<number, number> = {};
    active.forEach((t) => { pcCounts[t.pc] = (pcCounts[t.pc] || 0) + 1; });
    const majorityPC = Object.entries(pcCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
    this.threads.forEach((t) => {
      t.divergent = !t.done && t.pc !== parseInt(majorityPC ?? '0');
    });

    for (const t of this.threads) {
      if (!t.done) this.executeThread(t);
    }

    this.cycle++;
    return this.getState();
  }

  private executeThread(thread: ThreadState) {
    switch (thread.stage) {
      case PipelineStage.FETCH:     this.stageFetch(thread);     break;
      case PipelineStage.DECODE:    this.stageDecode(thread);    break;
      case PipelineStage.EXECUTE:   this.stageExecute(thread);   break;
      case PipelineStage.MEMORY:    this.stageMemory(thread);    break;
      case PipelineStage.WRITEBACK: this.stageWriteback(thread); break;
    }
  }

  private decoded = new Map<number, ReturnType<typeof decode32> & { raw: number }>();
  private threadKey(t: ThreadState) { return t.threadId + t.blockId * 1000; }

  // ── Pipeline stages ────────────────────────────────────────────────────

  private stageFetch(thread: ThreadState) {
    if (thread.pc >= this.program.length) {
      thread.done = true;
      thread.stage = PipelineStage.DONE;
      return;
    }
    const raw = this.program[thread.pc] >>> 0;
    const fields = decode32(raw);
    const opLabel = OPCODE_NAMES[fields.opcode] ?? `0x${fields.opcode.toString(16)}`;

    if (fields.opcode === RVOpcode.OP) {
      if (fields.funct7 === 0x01)
        thread.currentInstruction = fields.funct3 === 0x4 ? 'DIV' : 'MUL';
      else if (fields.funct7 === 0x20 && fields.funct3 === 0x0)
        thread.currentInstruction = 'SUB';
      else
        thread.currentInstruction = 'ADD';
    } else if (fields.opcode === RVOpcode.BRANCH) {
      const names = ['BEQ','BNE','???','???','BLT','BGE','BLTU','BGEU'];
      thread.currentInstruction = names[fields.funct3] ?? 'BR';
    } else if (fields.opcode === RVOpcode.CUSTOM0) {
      const names = ['MVM', 'SLDR', 'SSTR', 'CSET'];
      thread.currentInstruction = names[fields.funct3] ?? 'CUSTOM';
    } else if (fields.opcode === RVOpcode.CUSTOM1) {
      const names = ['XADD', 'XSUB', 'XMUL'];
      thread.currentInstruction = names[fields.funct3] ?? 'XARITH';
    } else {
      thread.currentInstruction = opLabel;
    }

    this.decoded.set(this.threadKey(thread), { ...fields, raw });
    thread.stage = PipelineStage.DECODE;
  }

  private stageDecode(thread: ThreadState) {
    thread.stage = PipelineStage.EXECUTE;
  }

  private stageExecute(thread: ThreadState) {
    const d = this.decoded.get(this.threadKey(thread));
    if (!d) { thread.stage = PipelineStage.FETCH; return; }
    const regs = thread.registers;

    switch (d.opcode) {
      case RVOpcode.OP: {
        let result = 0;
        if (d.funct7 === 0x01) {
          result = d.funct3 === 0x4
            ? (regs[d.rs2] !== 0 ? Math.trunc(regs[d.rs1] / regs[d.rs2]) | 0 : 0)
            : Math.imul(regs[d.rs1], regs[d.rs2]);
        } else if (d.funct7 === 0x20 && d.funct3 === 0x0) {
          result = (regs[d.rs1] - regs[d.rs2]) | 0;
        } else {
          switch (d.funct3) {
            case 0x0: result = (regs[d.rs1] + regs[d.rs2]) | 0; break;
            case 0x2: result = ((regs[d.rs1] | 0) < (regs[d.rs2] | 0)) ? 1 : 0; break;
            default:  result = 0;
          }
        }
        this.writeReg(thread, d.rd, result);
        thread.pc++;
        break;
      }
      case RVOpcode.OP_IMM: {
        let result = 0;
        switch (d.funct3) {
          case 0x0: result = (regs[d.rs1] + d.immI) | 0; break;
          default:  result = 0;
        }
        this.writeReg(thread, d.rd, result);
        thread.pc++;
        break;
      }
      case RVOpcode.LOAD:
      case RVOpcode.STORE:
      case RVOpcode.CUSTOM0:
      case RVOpcode.CUSTOM1:
        thread.stage = PipelineStage.MEMORY;
        return;
      case RVOpcode.BRANCH: {
        const r1 = regs[d.rs1] | 0;
        const r2 = regs[d.rs2] | 0;
        let taken = false;
        switch (d.funct3) {
          case 0x0: taken = r1 === r2; break;
          case 0x1: taken = r1 !== r2; break;
          case 0x4: taken = r1 < r2;   break;
          case 0x5: taken = r1 >= r2;  break;
        }
        thread.pc = taken
          ? Math.round((thread.pc * 4 + d.immB) / 4)
          : thread.pc + 1;
        break;
      }
      case RVOpcode.JAL: {
        const byteTarget = thread.pc * 4 + d.immJ;
        if (d.rd !== 0) this.writeReg(thread, d.rd, thread.pc + 1);
        thread.pc = Math.round(byteTarget / 4);
        break;
      }
      case RVOpcode.LUI:
        this.writeReg(thread, d.rd, d.raw & 0xFFFFF000);
        thread.pc++;
        break;
      case RVOpcode.MISC_MEM:  // FENCE → barrier
        thread.stage = PipelineStage.BARRIER;
        this.decoded.delete(this.threadKey(thread));
        return;
      case RVOpcode.SYSTEM:    // ECALL → thread done
        thread.done = true;
        thread.stage = PipelineStage.DONE;
        this.decoded.delete(this.threadKey(thread));
        return;
      default:
        thread.pc++;
    }

    this.decoded.delete(this.threadKey(thread));
    thread.stage = PipelineStage.WRITEBACK;
  }

  private stageMemory(thread: ThreadState) {
    const d = this.decoded.get(this.threadKey(thread));
    if (!d) { thread.stage = PipelineStage.FETCH; return; }
    const regs = thread.registers;

    switch (d.opcode) {
      case RVOpcode.LOAD: {
        const addr = (regs[d.rs1] + d.immI) & 0xFF;
        if (d.funct3 === 0x2)  // LW
          this.writeReg(thread, d.rd, this.memory[addr] & 0xFF);
        thread.pc++;
        break;
      }
      case RVOpcode.STORE: {
        const addr = (regs[d.rs1] + d.immS) & 0xFF;
        if (d.funct3 === 0x2)  // SW
          this.memory[addr] = regs[d.rs2] & 0xFF;
        thread.pc++;
        break;
      }
      case RVOpcode.CUSTOM0: {
        switch (d.funct3) {
          case 0x0: {
            // MVM — Analog matrix-vector multiply via Kirchhoff's current law
            // I_Q = Σ_P V_P · G_{P,Q}   (single analog step, ≈1 cycle)
            const srcAddr = regs[d.rs1] & 0xFF;
            const dstAddr = regs[d.rd]  & 0xFF;
            const result  = this.performMVM(srcAddr);
            for (let i = 0; i < 16 && (dstAddr + i) < 256; i++)
              this.memory[(dstAddr + i) & 0xFF] = result[i] & 0xFF;
            break;
          }
          case 0x1: {
            // SLDR — Scratchpad (shared) memory load
            const addr = regs[d.rs1] & 0x3F;
            this.writeReg(thread, d.rd, this.sharedMemory[addr] & 0xFF);
            break;
          }
          case 0x2: {
            // SSTR — Scratchpad (shared) memory store
            const addr = regs[d.rs1] & 0x3F;
            this.sharedMemory[addr] = regs[d.rs2] & 0xFF;
            break;
          }
          case 0x3: {
            // CSET — Program memristor crossbar cell
            // rd = value register (source), rs1 = row register, rs2 = col register
            // Executes: pot/dep pulse sequence at (row, col) to reach G(val).
            // Read voltage: ≤500mV (non-disturbing); write threshold: 830mV.
            const row = regs[d.rs1] & 0x3F;   // 0–63
            const col = regs[d.rs2] & 0x3F;   // 0–63
            const val = regs[d.rd];
            this.writeMemristorCell(
              row, col, val,
              col === 0 ? (RV_ABI[row] ?? `x${row}`) : `W[${row}][${col}]`
            );
            break;
          }
        }
        thread.pc++;
        break;
      }

      case RVOpcode.CUSTOM1: {
        // custom-1 (0x2B) — Crossbar-native integer arithmetic
        // Uses col 62 (XA_SCRATCH.COL) as scratch space.
        // Physical mechanism:
        //   XADD: conductance superposition — G_R = G_A + G_B − G_MIN
        //   XSUB: differential pair         — G_R = G_A − G_B + G_MIN
        //   XMUL: Ohm's law                 — I   = V(A) · G(B) → result
        const a = regs[d.rs1] | 0;
        const b = regs[d.rs2] | 0;
        const col = XA_SCRATCH.COL;
        const { XADD_ROW_A, XADD_ROW_B, XADD_ROW_R,
                XSUB_ROW_A, XSUB_ROW_B, XSUB_ROW_R,
                XMUL_ROW_V, XMUL_ROW_G, XMUL_ROW_R } = XA_SCRATCH;

        switch (d.funct3) {
          case 0x0: {
            // XADD — crossbar integer addition via conductance superposition
            // Program operand cells, combine at bit line, map back to integer
            this.writeMemristorCell(XADD_ROW_A, col, a & 0xFF, 'XADD.A');
            this.writeMemristorCell(XADD_ROW_B, col, b & 0xFF, 'XADD.B');
            // G_R = G(A) + G(B) − G_MIN  →  result = A + B (saturating at 255)
            const G_A = this.crossbar.conductances[XADD_ROW_A][col];
            const G_B = this.crossbar.conductances[XADD_ROW_B][col];
            const G_R_add = Math.min(G_MAX_US, G_A + G_B - G_MIN_US);
            const addResult = (a + b) | 0;   // exact RISC-V integer result
            this.writeMemristorCell(XADD_ROW_R, col, addResult & 0xFF, 'XADD.R');
            this.writeReg(thread, d.rd, addResult);
            break;
          }
          case 0x1: {
            // XSUB — crossbar integer subtraction via differential pair
            this.writeMemristorCell(XSUB_ROW_A, col, a & 0xFF, 'XSUB.A');
            this.writeMemristorCell(XSUB_ROW_B, col, b & 0xFF, 'XSUB.B');
            // G_R = G(A) − G(B) + G_MIN  →  result = A − B (signed)
            const G_Asub = this.crossbar.conductances[XSUB_ROW_A][col];
            const G_Bsub = this.crossbar.conductances[XSUB_ROW_B][col];
            const G_R_sub = Math.max(G_MIN_US, G_Asub - G_Bsub + G_MIN_US);
            const subResult = (a - b) | 0;   // exact RISC-V integer result
            this.writeMemristorCell(XSUB_ROW_R, col, (subResult & 0xFF + 256) & 0xFF, 'XSUB.R');
            this.writeReg(thread, d.rd, subResult);
            break;
          }
          case 0x2: {
            // XMUL — crossbar multiplication via Ohm's law: I = V · G
            // Program G-cell with factor B; apply voltage proportional to A; read current
            this.writeMemristorCell(XMUL_ROW_V, col, a & 0xFF, 'XMUL.V');
            this.writeMemristorCell(XMUL_ROW_G, col, b & 0xFF, 'XMUL.G');
            // I (µA) = V(A) · G(B)  where V(A) = (A/255) × READ_V
            const READ_V = MEMRISTOR_PHYSICS.READ_V_MV / 1000;
            const V_A    = ((a & 0xFF) / 255) * READ_V;
            const G_B    = this.crossbar.conductances[XMUL_ROW_G][col];
            const I_uA   = V_A * G_B;
            // Normalise analog result: I_MAX = READ_V * G_MAX
            const I_MAX  = READ_V * G_MAX_US;
            const analogResult = Math.round(I_uA / I_MAX * 255 * 255) & 0xFF;
            const mulResult  = Math.imul(a, b);   // exact RISC-V MUL (lower 32 bits)
            this.writeMemristorCell(XMUL_ROW_R, col, analogResult, 'XMUL.R');
            this.writeReg(thread, d.rd, mulResult);
            break;
          }
        }
        thread.pc++;
        break;
      }
    }

    this.decoded.delete(this.threadKey(thread));
    thread.stage = PipelineStage.WRITEBACK;
  }

  private stageWriteback(thread: ThreadState) {
    thread.stage = PipelineStage.FETCH;
  }

  // ── Analog MVM on the memristor crossbar ──────────────────────────────

  /**
   * Implements the VMM formula: I_Q = Σ_P V_P · G_{P,Q}
   *
   * Uses 16 input values from global memory as voltages (scaled to READ_V_MV).
   * Weight matrix is crossbar rows 0–15, cols 1–16.
   * Output is normalized per-column using differential conductance (G − G_MIN),
   * so unprogrammed cells (all at G_MIN) contribute zero.
   */
  private performMVM(srcAddr: number): number[] {
    const READ_V = READ_V_MV / 1000;  // 0.5 V
    const input  = Array.from({ length: 16 }, (_, p) => this.memory[(srcAddr + p) & 0xFF]);

    const result: number[] = [];
    for (let q = 0; q < 16; q++) {
      let diffCurrent_uA = 0;
      let colSumDiffG    = 0;

      for (let p = 0; p < 16; p++) {
        const V_p   = (input[p] / 255) * READ_V;
        const G_pq  = this.crossbar.conductances[p][q + 1];
        const G_diff = G_pq - G_MIN_US;          // differential (above baseline)
        if (G_diff > 0) {
          diffCurrent_uA += V_p * G_diff;
          colSumDiffG    += G_diff;
        }
      }

      // Normalize: output = weighted-average input value (0–255)
      // colSumDiffG * READ_V = max possible differential current for this column
      const colMaxI = colSumDiffG * READ_V;
      const normalized = colMaxI > 0
        ? Math.min(255, Math.max(0, Math.round(diffCurrent_uA / colMaxI * 255)))
        : 0;
      result.push(normalized);
    }

    this.crossbar.lastMVMResult = result;
    return result;
  }

  // ── Full run ───────────────────────────────────────────────────────────

  runToEnd(maxCycles = 10000): SimulationState[] {
    const history: SimulationState[] = [this.getState()];
    while (!this.isDone() && this.cycle < maxCycles) {
      history.push(this.step());
    }
    return history;
  }
}

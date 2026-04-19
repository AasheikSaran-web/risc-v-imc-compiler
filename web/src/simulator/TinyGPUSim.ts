// RISC-V + Memristor Crossbar In-Memory Compute simulator.
// Each thread runs a RISC-V RV32IM core with access to a shared 16×16 memristor crossbar.
// Custom-0 opcode (0x0B) dispatches MVM/SLDR/SSTR to the crossbar and scratchpad.

import {
  Instruction,
  RVOpcode,
  OPCODE_NAMES,
  PipelineStage,
  ThreadState,
  SimulationState,
  CrossbarState,
} from '../compiler/types';

// Sign-extend a value of `bits` width
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

  // I-type immediate [31:20]
  const immI = signExt((instr >>> 20), 12);

  // S-type immediate [31:25|11:7]
  const immS = signExt(((instr >>> 25) << 5) | ((instr >> 7) & 0x1F), 12);

  // B-type immediate [31|7|30:25|11:8] → sign-extended 13-bit
  const immB = signExt(
    (((instr >>> 31) & 1) << 12) |
    (((instr >> 7)  & 1) << 11) |
    (((instr >>> 25) & 0x3F) << 5) |
    (((instr >> 8)  & 0xF) << 1),
    13
  );

  // J-type immediate [31|19:12|20|30:21] → sign-extended 21-bit
  const immJ = signExt(
    (((instr >>> 31) & 1) << 20) |
    (((instr >> 12) & 0xFF) << 12) |
    (((instr >>> 20) & 1) << 11) |
    (((instr >>> 21) & 0x3FF) << 1),
    21
  );

  return { opcode, rd, funct3, rs1, rs2, funct7, immI, immS, immB, immJ };
}

/** RISC-V + Memristor Crossbar In-Memory Compute simulator */
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

    // Parse 32-bit hex instructions
    this.program = instructions.map((i) => parseInt(i.hex.replace('0x', ''), 16) >>> 0);

    this.memory = new Array(256).fill(0);
    for (let i = 0; i < initialMemory.length && i < 256; i++) {
      this.memory[i] = initialMemory[i] & 0xFF;
    }

    this.sharedMemory = new Array(64).fill(0);

    // Initialize crossbar: identity-like diagonal for demo (each output = corresponding input)
    const conductances: number[][] = Array.from({ length: 16 }, (_, row) =>
      Array.from({ length: 16 }, (_, col) => (row === col ? 128 : 16))
    );
    this.crossbar = { conductances, lastMVMResult: new Array(16).fill(0) };

    this.initBlock(0);
  }

  private initBlock(blockId: number) {
    this.currentBlock = blockId;
    this.threads = [];
    this.sharedMemory = new Array(64).fill(0);

    for (let t = 0; t < this.threadsPerBlock; t++) {
      // x0=0, x25=threadIdx, x26=blockIdx, x27=blockDim
      const regs = new Array(32).fill(0);
      regs[25] = t;         // s9  = threadIdx
      regs[26] = blockId;   // s10 = blockIdx
      regs[27] = this.threadsPerBlock; // s11 = blockDim
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
    }
  }

  getState(): SimulationState {
    return {
      cycle: this.cycle,
      threads: this.threads.map((t) => ({ ...t, registers: [...t.registers] })),
      memory: [...this.memory],
      sharedMemory: [...this.sharedMemory],
      crossbar: {
        conductances: this.crossbar.conductances.map(row => [...row]),
        lastMVMResult: [...this.crossbar.lastMVMResult],
      },
      currentBlock: this.currentBlock,
      totalBlocks: this.numBlocks,
    };
  }

  isDone(): boolean {
    return this.currentBlock >= this.numBlocks;
  }

  step(): SimulationState {
    if (this.isDone()) return this.getState();

    if (this.threads.every((t) => t.done)) {
      this.currentBlock++;
      if (this.currentBlock < this.numBlocks) {
        this.initBlock(this.currentBlock);
      }
      this.cycle++;
      return this.getState();
    }

    // Barrier: hold all until everyone arrives
    const atBarrier = this.threads.filter(t => !t.done && t.stage === PipelineStage.BARRIER);
    if (atBarrier.length > 0) {
      const activeThreads = this.threads.filter(t => !t.done);
      if (atBarrier.length === activeThreads.length) {
        for (const thread of atBarrier) {
          thread.stage = PipelineStage.FETCH;
          thread.pc++;
        }
        this.cycle++;
        return this.getState();
      }
      for (const thread of this.threads) {
        if (thread.done || thread.stage === PipelineStage.BARRIER) continue;
        this.executeThread(thread);
      }
      this.cycle++;
      return this.getState();
    }

    // Divergence tracking
    const activePCs = new Set(this.threads.filter(t => !t.done).map(t => t.pc));
    if (activePCs.size > 1) {
      const pcCounts: Record<number, number> = {};
      this.threads.filter(t => !t.done).forEach(t => {
        pcCounts[t.pc] = (pcCounts[t.pc] || 0) + 1;
      });
      const majorityPC = Object.entries(pcCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
      this.threads.forEach(t => {
        if (!t.done) t.divergent = t.pc !== parseInt(majorityPC ?? '0');
      });
    } else {
      this.threads.forEach(t => { t.divergent = false; });
    }

    for (const thread of this.threads) {
      if (thread.done) continue;
      this.executeThread(thread);
    }

    this.cycle++;
    return this.getState();
  }

  private executeThread(thread: ThreadState) {
    switch (thread.stage) {
      case PipelineStage.FETCH:     this.stageFetch(thread); break;
      case PipelineStage.DECODE:    this.stageDecode(thread); break;
      case PipelineStage.EXECUTE:   this.stageExecute(thread); break;
      case PipelineStage.MEMORY:    this.stageMemory(thread); break;
      case PipelineStage.WRITEBACK: this.stageWriteback(thread); break;
    }
  }

  // Decoded instruction cache keyed by thread identity
  private decoded = new Map<number, ReturnType<typeof decode32> & { raw: number }>();

  private threadKey(t: ThreadState) { return t.threadId + t.blockId * 1000; }

  private stageFetch(thread: ThreadState) {
    if (thread.pc >= this.program.length) {
      thread.done = true;
      thread.stage = PipelineStage.DONE;
      return;
    }
    const raw = this.program[thread.pc] >>> 0;
    const fields = decode32(raw);
    const opLabel = OPCODE_NAMES[fields.opcode] ?? `0x${fields.opcode.toString(16)}`;
    // For R-type, refine label using funct3/funct7
    if (fields.opcode === RVOpcode.OP) {
      const f7 = fields.funct7;
      const f3 = fields.funct3;
      if (f7 === 0x01) thread.currentInstruction = f3 === 0x4 ? 'DIV' : 'MUL';
      else if (f7 === 0x20 && f3 === 0x0) thread.currentInstruction = 'SUB';
      else thread.currentInstruction = 'ADD';
    } else if (fields.opcode === RVOpcode.BRANCH) {
      const names = ['BEQ','BNE','???','???','BLT','BGE','BLTU','BGEU'];
      thread.currentInstruction = names[fields.funct3] ?? 'BR';
    } else if (fields.opcode === RVOpcode.CUSTOM0) {
      const names = ['MVM', 'SLDR', 'SSTR'];
      thread.currentInstruction = names[fields.funct3] ?? 'CUSTOM';
    } else {
      thread.currentInstruction = opLabel;
    }
    this.decoded.set(this.threadKey(thread), { ...fields, raw });
    thread.stage = PipelineStage.DECODE;
  }

  private stageDecode(thread: ThreadState) {
    // Decode is folded into fetch in this simplified model
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
            case 0x2: result = ((regs[d.rs1] | 0) < (regs[d.rs2] | 0)) ? 1 : 0; break; // SLT
            default:  result = 0;
          }
        }
        if (d.rd !== 0) regs[d.rd] = result;
        thread.pc++;
        break;
      }
      case RVOpcode.OP_IMM: {
        let result = 0;
        switch (d.funct3) {
          case 0x0: result = (regs[d.rs1] + d.immI) | 0; break; // ADDI
          default:  result = 0;
        }
        if (d.rd !== 0) regs[d.rd] = result;
        thread.pc++;
        break;
      }
      case RVOpcode.LOAD:
        thread.stage = PipelineStage.MEMORY;
        return;
      case RVOpcode.STORE:
        thread.stage = PipelineStage.MEMORY;
        return;
      case RVOpcode.CUSTOM0:
        thread.stage = PipelineStage.MEMORY;
        return;
      case RVOpcode.BRANCH: {
        const r1 = regs[d.rs1] | 0;
        const r2 = regs[d.rs2] | 0;
        let taken = false;
        switch (d.funct3) {
          case 0x0: taken = r1 === r2; break;  // BEQ
          case 0x1: taken = r1 !== r2; break;  // BNE
          case 0x4: taken = r1 < r2; break;    // BLT
          case 0x5: taken = r1 >= r2; break;   // BGE
        }
        if (taken) {
          const byteTarget = thread.pc * 4 + d.immB;
          thread.pc = Math.round(byteTarget / 4);
        } else {
          thread.pc++;
        }
        break;
      }
      case RVOpcode.JAL: {
        const byteTarget = thread.pc * 4 + d.immJ;
        if (d.rd !== 0) regs[d.rd] = thread.pc + 1;
        thread.pc = Math.round(byteTarget / 4);
        break;
      }
      case RVOpcode.LUI:
        if (d.rd !== 0) regs[d.rd] = d.raw & 0xFFFFF000;
        thread.pc++;
        break;
      case RVOpcode.MISC_MEM: // FENCE → barrier
        thread.stage = PipelineStage.BARRIER;
        this.decoded.delete(this.threadKey(thread));
        return;
      case RVOpcode.SYSTEM: // ECALL → thread done
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
        if (d.funct3 === 0x2 && d.rd !== 0) { // LW
          regs[d.rd] = this.memory[addr] & 0xFF;
        }
        thread.pc++;
        break;
      }
      case RVOpcode.STORE: {
        const addr = (regs[d.rs1] + d.immS) & 0xFF;
        if (d.funct3 === 0x2) { // SW
          this.memory[addr] = regs[d.rs2] & 0xFF;
        }
        thread.pc++;
        break;
      }
      case RVOpcode.CUSTOM0: {
        switch (d.funct3) {
          case 0x0: { // MVM — Matrix-Vector Multiply on crossbar
            const srcAddr = regs[d.rs1] & 0xFF;
            const dstAddr = regs[d.rd] & 0xFF;
            const result = this.performMVM(srcAddr);
            for (let i = 0; i < 16 && (dstAddr + i) < 256; i++) {
              this.memory[(dstAddr + i) & 0xFF] = result[i] & 0xFF;
            }
            break;
          }
          case 0x1: { // SLDR — Shared (scratchpad) memory load
            const addr = regs[d.rs1] & 0x3F;
            if (d.rd !== 0) regs[d.rd] = this.sharedMemory[addr] & 0xFF;
            break;
          }
          case 0x2: { // SSTR — Shared (scratchpad) memory store
            const addr = regs[d.rs1] & 0x3F;
            this.sharedMemory[addr] = regs[d.rs2] & 0xFF;
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
    // Register writeback is performed inline in execute/memory for simplicity
    thread.stage = PipelineStage.FETCH;
  }

  /** Analog matrix-vector multiply on the memristor crossbar.
   *  output[row] = Σ_col (conductance[row][col] * input[col]) >> 8  (normalized)
   */
  private performMVM(srcAddr: number): number[] {
    const input = Array.from({ length: 16 }, (_, col) => this.memory[(srcAddr + col) & 0xFF]);
    const result = this.crossbar.conductances.map((row) => {
      const sum = row.reduce((acc, g, col) => acc + g * input[col], 0);
      return Math.min(255, Math.max(0, (sum >> 8) & 0xFF));
    });
    this.crossbar.lastMVMResult = result;
    return result;
  }

  runToEnd(maxCycles = 10000): SimulationState[] {
    const history: SimulationState[] = [this.getState()];
    while (!this.isDone() && this.cycle < maxCycles) {
      history.push(this.step());
    }
    return history;
  }
}

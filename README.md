# RISC-V IMC Compiler

**An MLIR-based compiler that lowers parallel kernels to RISC-V instructions executing on a memristor crossbar — making In-Memory Compute visible, step by step.**

Most engineers know that matrix operations are the bottleneck in modern ML workloads. Few understand how a compiler bridges high-level code and analog hardware that computes inside memory itself. This project makes that journey visible. Write a kernel in a C-like DSL, watch it lower through RISC-V intermediate representations, see 32-bit binary instructions get emitted, and then simulate those instructions running on a memristor crossbar — all in your browser.

Built on [MLIR](https://mlir.llvm.org/), targeting a **RISC-V RV32IM** core augmented with a custom **memristor crossbar In-Memory Compute** extension.

**By [Aasheik Saran](https://github.com/AasheikSaran-web)**

---

## What Is In-Memory Compute?

Traditional von Neumann architecture suffers from the **memory wall**: data must travel from memory to the processor and back for every operation. For matrix-heavy workloads (neural networks, scientific computing), this data movement dominates energy and latency.

**In-Memory Compute (IMC)** eliminates this bottleneck by performing computation *inside* the memory array itself, exploiting the physics of the storage devices.

### The Memristor Crossbar

A memristor is a two-terminal device whose resistance (and therefore conductance) depends on the history of current that has passed through it. It can be programmed to hold an analog value — making it ideal as a synaptic weight.

Arranged in a **crossbar array**, memristors enable matrix-vector multiplication in a single analog step:

```
         Input voltages V₀  V₁  V₂  …  Vₙ
                         │   │   │       │
          ┌──────────────┼───┼───┼───────┼──────────────┐
Row 0 ───┤  G₀₀  G₀₁  G₀₂     G₀ₙ  ├── I₀ = Σ Gᵢⱼ·Vⱼ
Row 1 ───┤  G₁₀  G₁₁  G₁₂     G₁ₙ  ├── I₁
Row 2 ───┤  G₂₀  G₂₁  G₂₂     G₂ₙ  ├── I₂
  …      │                           │    …
Row m ───┤  Gₘ₀  Gₘ₁  Gₘ₂     Gₘₙ  ├── Iₘ
          └──────────────────────────────────┘
                                          Output currents
                                    (Kirchhoff's current law)
```

By Kirchhoff's current law, the output current at each row wire is the **dot product** of the input voltage vector and the memristor conductance row. The entire **matrix-vector multiply** completes in a single analog step — O(1) time, regardless of matrix size — at a fraction of the energy of digital computation.

### Why RISC-V?

RISC-V is an open, royalty-free ISA designed for extensibility. Its **custom opcode space** (custom-0 through custom-3) lets hardware architects add domain-specific instructions without breaking standard toolchains. This compiler uses **custom-0 (opcode `0x0B`)** for memristor crossbar dispatch instructions, showing exactly how a real IMC accelerator would extend a RISC-V core.

---

## Compilation Pipeline

```
     .tgc Source Code
            │
            ▼
   ┌──────────────────┐
   │  Lexer + Parser  │   Tokenizes source, builds AST via recursive
   └────────┬─────────┘   descent; recognizes mvm() crossbar calls
            │
            ▼
   ┌──────────────────┐
   │  RISC-V IMC      │   Walks AST, emits riscv-imc dialect ops;
   │  IR Generation   │   maps pointer params to 64-byte memory regions
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Optimization    │   Constant propagation, register reuse,
   │  Passes          │   dead-code elimination
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Register Alloc  │   Linear scan over x5–x24 (20 GPRs).
   └────────┬─────────┘   x25/x26/x27 reserved for tid/bid/bdm
            │
            ▼
   ┌──────────────────┐
   │  Binary Emitter  │   Encodes each op into a 32-bit RISC-V word;
   └────────┬─────────┘   custom-0 words for MVM / SLDR / SSTR
            │
            ▼
    32-bit RISC-V Binary
    (runs on RISC-V core +
     memristor crossbar IMC)
```

### Example: Vector Add

```c
// Source (.tgc)
kernel vector_add(global int* a, global int* b, global int* c) {
    int i = blockIdx * blockDim + threadIdx;
    c[i] = a[i] + b[i];
}
```

```
// RISC-V IMC IR (riscv-imc dialect)
riscv-imc.func @vector_add() {
  %0 = riscv.block_id          // bid  = x26
  %1 = riscv.block_dim         // bdm  = x27
  %2 = riscv.mul %0, %1        // MUL  t0, bid, bdm
  %3 = riscv.thread_id         // tid  = x25
  %4 = riscv.add %2, %3        // ADD  t0, t0, tid
  %5 = riscv.lw [%4]           // LW   t1, 0(t0)    ; a[i]
  ...
  riscv.ecall                  // ECALL             ; thread complete
}
```

```asm
; 32-bit RISC-V assembly output
0x00000013  ADDI t0, x0, 0          ; i = 0
0x01B28233  MUL  t0, s10, s11       ; i = blockIdx * blockDim
0x01928233  ADD  t0, t0, s9         ; i += threadIdx
0x00028283  LW   t1, 0(t0)          ; t1 = a[i]
...
0x00000073  ECALL                   ; thread done
```

---

## RISC-V ISA

This compiler targets **RISC-V RV32IM** — the 32-bit base integer ISA plus the integer Multiply/Divide extension — augmented with a custom memristor crossbar extension in the **custom-0** opcode space.

### Instruction Formats (32-bit)

All RISC-V instructions are exactly 32 bits wide. There are six canonical formats:

```
R-type  [31:25 funct7][24:20 rs2][19:15 rs1][14:12 funct3][11:7 rd][6:0 opcode]
I-type  [31:20 imm[11:0]        ][19:15 rs1][14:12 funct3][11:7 rd][6:0 opcode]
S-type  [31:25 imm[11:5]][24:20 rs2][19:15 rs1][14:12 funct3][11:7 imm[4:0]][6:0 opcode]
B-type  [31 imm[12]][30:25 imm[10:5]][24:20 rs2][19:15 rs1][14:12 funct3][11:8 imm[4:1]][7 imm[11]][6:0 opcode]
U-type  [31:12 imm[31:12]                                  ][11:7 rd][6:0 opcode]
J-type  [31 imm[20]][30:21 imm[10:1]][20 imm[11]][19:12 imm[19:12]][11:7 rd][6:0 opcode]
```

### Base Instructions (RV32IM)

| Opcode | Format | Mnemonic | Operation |
|--------|--------|----------|-----------|
| `0110011` | R | **ADD** rd, rs1, rs2 | rd = rs1 + rs2 |
| `0110011` | R | **SUB** rd, rs1, rs2 | rd = rs1 − rs2 (funct7=0x20) |
| `0110011` | R | **MUL** rd, rs1, rs2 | rd = rs1 × rs2 (funct7=0x01, M-ext) |
| `0110011` | R | **DIV** rd, rs1, rs2 | rd = rs1 ÷ rs2 (funct7=0x01, M-ext) |
| `0010011` | I | **ADDI** rd, rs1, imm | rd = rs1 + imm (load immediate) |
| `0000011` | I | **LW** rd, imm(rs1) | rd = mem[rs1 + imm] (global load) |
| `0100011` | S | **SW** rs2, imm(rs1) | mem[rs1 + imm] = rs2 (global store) |
| `1100011` | B | **BEQ** rs1, rs2, off | if rs1 == rs2: PC += off |
| `1100011` | B | **BNE** rs1, rs2, off | if rs1 != rs2: PC += off |
| `1100011` | B | **BLT** rs1, rs2, off | if rs1 < rs2: PC += off (signed) |
| `1100011` | B | **BGE** rs1, rs2, off | if rs1 ≥ rs2: PC += off (signed) |
| `1101111` | J | **JAL** rd, off | rd = PC+4; PC += off (unconditional jump) |
| `0110111` | U | **LUI** rd, imm | rd = imm << 12 (load upper immediate) |
| `0001111` | — | **FENCE** | Memory/thread barrier (`__syncthreads`) |
| `1110011` | — | **ECALL** | System call — signals thread completion |

### Custom Memristor Crossbar Extension (custom-0, opcode `0x0B`)

The crossbar instructions occupy RISC-V's custom-0 opcode space, using `funct3` to distinguish operations:

| funct3 | Mnemonic | Operation |
|--------|----------|-----------|
| `000` | **MVM** rd, rs1 | Crossbar matrix-vector multiply. Reads 16-element input vector from `mem[rs1]`, multiplies by the 16×16 conductance matrix, writes 16-element result to `mem[rd]`. Single analog step ≈ 1 cycle. |
| `001` | **SLDR** rd, rs1 | Scratchpad (shared) memory load: `rd = scratchpad[rs1]` |
| `010` | **SSTR** rs1, rs2 | Scratchpad (shared) memory store: `scratchpad[rs1] = rs2` |

**MVM binary encoding (R-type layout, custom-0):**
```
  31      25 24   20 19   15 14  12 11    7 6      0
 ┌──────────┬───────┬───────┬──────┬───────┬────────┐
 │ 0000000  │ 00000 │  rs1  │ 000  │  rd   │0001011 │
 └──────────┴───────┴───────┴──────┴───────┴────────┘
   funct7     rs2    rs1   funct3    rd     opcode
```

### Register File (x0–x31, RISC-V ABI)

| Registers | ABI Name | Role in this compiler |
|-----------|----------|-----------------------|
| x0 | `zero` | Hardwired 0 — used for immediate loads via ADDI |
| x5–x7 | `t0–t2` | Allocatable temporaries |
| x8–x9 | `s0–s1` | Allocatable saved registers |
| x10–x17 | `a0–a7` | Allocatable argument/result registers |
| x18–x24 | `s2–s8` | Allocatable saved registers |
| x25 | `s9` / `tid` | **threadIdx** — thread index within block (read-only) |
| x26 | `s10` / `bid` | **blockIdx** — current block index (read-only) |
| x27 | `s11` / `bdm` | **blockDim** — threads per block (read-only) |
| x28–x31 | `t3–t6` | Reserved |

The linear scan allocator manages **20 general-purpose registers** (x5–x24).

---

## Memristor Crossbar Hardware

### Physical Model

Each memristor in the crossbar is a two-terminal resistive device characterized by its **conductance** G (in units normalised to 0–255 in the simulator). The crossbar implements:

```
I_row = Σ_col ( G[row][col] × V_col )
```

where:
- **V_col** — input voltage applied to column wire (proportional to input data)
- **G[row][col]** — conductance of the memristor at position (row, col), programmed as a weight
- **I_row** — output current on row wire (read by ADC, proportional to dot product result)

This is physically equivalent to a **matrix-vector multiply** performed in the analog domain by Kirchhoff's current law — no separate compute unit needed.

### Crossbar Simulator (16×16)

The simulator models a 16×16 memristor crossbar:

```
Crossbar state: G[16][16]   (conductance values, 0–255)
MVM operation:
  for row in 0..15:
    I[row] = Σ_col(G[row][col] × mem[srcAddr + col]) >> 8
  mem[dstAddr .. dstAddr+15] = I[0..15]
```

The `>> 8` normalises the accumulated product to 8-bit range. The crossbar conductance grid is visualised live in the simulator panel.

### Performance Advantage

| Operation | Scalar RISC-V (digital) | Crossbar MVM (analog IMC) |
|-----------|-------------------------|--------------------------|
| 16×16 MatVec | 256 MUL + 255 ADD = 511 ops | **1 MVM** (single analog step) |
| Estimated cycles | ~2560 (5-stage pipeline) | **~1 cycle** |
| Energy model | O(N²) operations | O(1) operations |

Each `MVM` instruction in this compiler saves approximately **511 scalar MACs** — shown live in the Analysis panel.

---

## The DSL

A minimal C-like language for expressing parallel kernels, with a crossbar extension:

```c
// Standard parallel kernel
kernel vector_add(global int* a, global int* b, global int* c) {
    int i = blockIdx * blockDim + threadIdx;
    c[i] = a[i] + b[i];
}
```

```c
// In-Memory Compute kernel using crossbar MVM
kernel matmul_imc(global int* input, global int* output) {
    // Crossbar performs the full 16x16 matrix-vector multiply in one step
    mvm(output, input);
}
```

```c
// Scratchpad memory with barrier synchronization
kernel shared_reduce(global int* input, global int* output) {
    shared int scratch[4];
    int idx = blockIdx * blockDim + threadIdx;
    scratch[threadIdx] = input[idx];
    __syncthreads();
    if (threadIdx < 2) {
        scratch[threadIdx] = scratch[threadIdx] + scratch[threadIdx + 2];
    }
    __syncthreads();
    if (threadIdx < 1) {
        output[blockIdx] = scratch[0];
    }
}
```

### Language Reference

| Feature | Syntax | Compiled To |
|---------|--------|-------------|
| Kernel declaration | `kernel name(params) { }` | Function entry point |
| Global pointer | `global int* name` | Memory region base address |
| Scalar parameter | `int name` | Loaded from address 192+ |
| Thread index | `threadIdx` | x25 (s9/tid) |
| Block index | `blockIdx` | x26 (s10/bid) |
| Threads per block | `blockDim` | x27 (s11/bdm) |
| Arithmetic | `+`, `-`, `*`, `/` | ADD/SUB/MUL/DIV |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` | BEQ/BNE/BLT/BGE |
| For loop | `for (int i = 0; i < n; i = i + 1)` | Branch + JAL |
| Conditional | `if (cond) { } else { }` | BEQ/BNE/BLT/BGE |
| Global load | `a[i]` | LW rd, 0(rs1) |
| Global store | `a[i] = v` | SW rs2, 0(rs1) |
| Scratchpad array | `shared int arr[size]` | Scratchpad allocation |
| Scratchpad load | `arr[i]` (on shared array) | SLDR rd, rs1 (custom-0) |
| Scratchpad store | `arr[i] = v` | SSTR rs1, rs2 (custom-0) |
| Thread barrier | `__syncthreads()` | FENCE |
| **Crossbar MVM** | `mvm(output, input)` | **MVM rd, rs1 (custom-0)** |

### Memory Layout

| Memory Type | Size | Latency | Scope |
|-------------|------|---------|-------|
| **Global** (DRAM) | 256 bytes | High (~4 cycles) | All blocks |
| **Scratchpad** (SRAM) | 64 bytes per block | Low (~1 cycle) | Single block |
| **Crossbar** (analog) | 16×16 conductances | **~1 cycle** for full MatVec | Shared |

Pointer parameters map to contiguous 64-byte regions in global memory:

| Parameter order | Address range |
|-----------------|--------------|
| 1st `global int*` | 0–63 |
| 2nd `global int*` | 64–127 |
| 3rd `global int*` | 128–191 |
| Scalar `int` | 192+ |

---

## Web Visualizer Features

| Panel | What It Shows |
|-------|---------------|
| **Source Editor** | Monaco editor with live compilation (300ms debounce) |
| **RISC-V IMC Pipeline** | Three compilation stages: IR generation → optimization → register allocation |
| **32-bit Binary** | Color-coded instruction fields: imm/rs2 · rs1 · funct3 · rd · opcode |
| **IMC Execution** | Cycle-by-cycle RISC-V + crossbar simulation; thread cards show 5-stage pipeline state |
| **Crossbar Grid** | Live 8×8 heatmap of memristor conductances; last MVM output vector |
| **Analysis** | Divergence, memory access patterns, crossbar utilization, cycles saved vs. scalar |

### Running the Visualizer

```bash
git clone https://github.com/AasheikSaran-web/risc-v-imc-compiler
cd risc-v-imc-compiler/web
npm install
npm run dev
# Open http://localhost:5173
```

---

## Examples

| Kernel | Description | Highlights |
|--------|-------------|------------|
| `vector_add` | `c[i] = a[i] + b[i]` | Basic RISC-V parallel kernel |
| `matrix_multiply` | Loop-based matmul | For-loops, register pressure |
| `dot_product` | `Σ a[i]·b[i]` | Multiply-accumulate pattern |
| `saxpy` | `y[i] = α·x[i] + y[i]` | Scalar parameter loading |
| `relu` | `max(0, x[i])` | Branch divergence analysis |
| `shared_reduce` | Parallel tree reduction | Scratchpad + FENCE barrier |
| `shared_tile_add` | Tiled addition | Scratchpad tiling pattern |
| `conv1d` | 1D sliding window | Nested loops, accumulation |
| `vector_max` | `max(a[i], b[i])` | Divergent branches |

---

## Project Structure

```
risc-v-imc-compiler/
  web/
    src/
      compiler/
        TGCCompiler.ts     # RISC-V IMC compiler (Lexer → Parser → IR → RegAlloc → Binary)
        types.ts           # Shared types: Instruction, CrossbarState, SimulationState, …
      simulator/
        TinyGPUSim.ts      # RISC-V + memristor crossbar cycle-accurate simulator
      components/
        Editor.tsx         # Monaco source editor
        PipelineView.tsx   # Compilation stage viewer
        BinaryView.tsx     # 32-bit RISC-V binary with field color-coding
        GPUSimulator.tsx   # IMC execution panel + crossbar grid + register file
        AnalysisPanel.tsx  # Divergence, coalescing, IMC efficiency analysis
      examples/index.ts    # Pre-loaded example kernels
  include/                 # C++ MLIR dialect headers (TinyGPU dialect, original)
  lib/                     # C++ MLIR dialect implementations
  tools/tgc/               # Command-line compiler driver
  examples/                # .tgc source kernels
  Dockerfile               # Reproducible LLVM/MLIR build
```

---

## How This Relates to Real IMC Compilers

| Concept | This Project | Production IMC (e.g., IBM AIHWKit, Analog AI) |
|---------|--------------|-----------------------------------------------|
| **IR** | RISC-V IMC dialect (MLIR) | ONNX / Torch FX → hardware IR |
| **Crossbar dispatch** | custom-0 MVM instruction | Proprietary crossbar API |
| **Weight mapping** | Identity/demo conductances | Quantization-aware training |
| **ADC/DAC model** | 8-bit normalization (>>8) | Full ADC non-linearity model |
| **Register allocation** | Linear scan, 20 GPRs | LLVM allocator with spill |
| **Tile size** | Fixed 16×16 | Variable (128×128 – 1K×1K) |
| **Noise model** | None (ideal) | Shot noise, conductance drift |
| **Memory hierarchy** | Global + scratchpad | HBM + SRAM tile buffers |

The fundamental idea — lowering matrix operations to analog crossbar dispatch — is identical. The simplifications make each step teachable without losing the essential architecture.

---

## Roadmap

- [x] RISC-V RV32IM backend (32-bit instruction encoding)
- [x] Memristor crossbar custom-0 extension (MVM / SLDR / SSTR)
- [x] RISC-V ABI register allocation (x5–x24)
- [x] 5-stage RISC-V pipeline simulator
- [x] 16×16 crossbar simulation with Kirchhoff model
- [x] `mvm()` DSL keyword for crossbar dispatch
- [x] Scratchpad memory + `__syncthreads()` (FENCE)
- [x] Thread divergence and memory coalescing analysis
- [x] Crossbar conductance heatmap visualization
- [x] IMC efficiency metrics (cycles saved vs. scalar)
- [ ] Conductance noise model (shot noise, retention drift)
- [ ] Weight quantization pass (map float weights to 8-bit conductances)
- [ ] Variable crossbar tile size (configurable N×N)
- [ ] WASM backend (run C++ MLIR compiler in browser)
- [ ] Hardware RTL generation (RISC-V core + crossbar controller in CIRCT)

---

## License

Apache 2.0 with LLVM Exceptions. See [LICENSE](LICENSE).

# RISC-V IMC Compiler

**An MLIR-based compiler that lowers parallel kernels to RISC-V instructions executing on a 64×64 molecular memristor crossbar — making In-Memory Compute visible, step by step.**

Write a kernel in a C-like DSL, watch it lower through RISC-V intermediate representations, see 32-bit binary instructions get emitted, and then simulate those instructions running on a memristor crossbar — all in your browser.

Built on [MLIR](https://mlir.llvm.org/), targeting **RISC-V RV32IM** augmented with two custom crossbar extensions: **custom-0** (MVM / SLDR / SSTR / CSET) and **custom-1** (XADD / XSUB / XMUL).

**By [Aasheik Saran](https://github.com/AasheikSaran-web)**

---

## What Is In-Memory Compute?

Traditional von Neumann architecture suffers from the **memory wall**: data must travel from memory to the processor and back for every operation. For matrix-heavy workloads (neural networks, scientific computing), this data movement dominates energy and latency.

**In-Memory Compute (IMC)** eliminates this bottleneck by performing computation *inside* the memory array itself, exploiting the physics of the storage devices.

### The Memristor Crossbar

A memristor is a two-terminal device whose conductance depends on the history of current through it. Programmed via voltage pulses, it can hold an analog value — making it ideal as both a weight and a compute element.

Arranged in a **64×64 crossbar array**, memristors enable matrix-vector multiplication in a single analog step:

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

The entire **matrix-vector multiply** — O(N²) scalar operations — completes in a **single analog step** regardless of matrix size.

### Physical Memristor Model

This compiler integrates the device model from:

> **"Linear symmetric self-selecting 14-bit kinetic molecular memristors"**  
> *Nature* **633**, Sep 2024 — [RuIIL](BF₄)₂ on a 64×64 crossbar

| Parameter | Value |
|-----------|-------|
| Conductance range | 200 nS – 5.9 mS (G_MIN – G_MAX) |
| Analog levels | 16,520 (≈14-bit resolution) |
| G step | ≈ 0.357 µS/level |
| Potentiation pulse | 900 mV / 80 ns |
| Depression pulse | −750 mV / 65 ns |
| Write threshold | 830 mV |
| Read voltage | ≤ 500 mV (non-disturbing) |
| Half-select voltage | 450 mV (below V_th, no switching) |
| Write accuracy (RMSE) | < 42 nS |
| Signal-to-noise ratio | 73–79 dB |

### Register File Backed by Memristors

Every RISC-V register write (`ADD`, `LW`, `ADDI`, etc.) **internally programs a memristor cell**:

```
RISC-V:   t0 = a[i]   (LW t0, 0(t1))
                │
                ▼  [pot/dep pulses @ 900mV/80ns or 750mV/65ns]
         crossbar[5][0] = G(value)     ← row 5 = x5 (t0), col 0 = register file
```

- **Column 0** — register file backing (rows 0–31 store conductance encoding of x0–x31)
- **Columns 1–63** — weight matrix for MVM operations
- **Column 62** — scratch region for XADD / XSUB / XMUL (rows 32–40)

The visualizer shows every register write as a pot/dep pulse event, with the resulting conductance in µS.

### Why RISC-V?

RISC-V is an open, royalty-free ISA with a reserved **custom opcode space** (custom-0 through custom-3) for domain-specific extensions. This compiler uses:

- **custom-0 (`0x0B`)** — crossbar matrix/memory operations (MVM, SLDR, SSTR, CSET)
- **custom-1 (`0x2B`)** — crossbar integer arithmetic (XADD, XSUB, XMUL)

---

## Compilation Pipeline

```
     .tgc Source Code
            │
            ▼
   ┌──────────────────┐
   │  Lexer + Parser  │   Tokenizes source, builds AST via recursive descent.
   └────────┬─────────┘   Recognizes mvm(), cset(), xadd(), xsub(), xmul().
            │
            ▼
   ┌──────────────────┐
   │  RISC-V IMC      │   Walks AST, emits riscv-imc dialect ops.
   │  IR Generation   │   Maps pointer params to 64-byte memory regions.
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Optimization    │   Constant propagation, register reuse,
   │  Passes          │   dead-code elimination.
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Register Alloc  │   Linear scan over x5–x24 (20 GPRs).
   └────────┬─────────┘   x25/x26/x27 reserved for tid/bid/bdm.
            │
            ▼
   ┌──────────────────┐
   │  Binary Emitter  │   Encodes each op into a 32-bit RISC-V word.
   └────────┬─────────┘   custom-0 and custom-1 words for IMC instructions.
            │
            ▼
    32-bit RISC-V Binary
    (runs on RISC-V core +
     64×64 memristor crossbar)
```

---

## RISC-V ISA

This compiler targets **RISC-V RV32IM** augmented with two custom memristor crossbar extensions.

### Instruction Formats (32-bit)

All RISC-V instructions are exactly 32 bits wide:

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
| `0110011` | R | **MUL** rd, rs1, rs2 | rd = rs1 × rs2 (M-ext, funct7=0x01) |
| `0110011` | R | **DIV** rd, rs1, rs2 | rd = rs1 ÷ rs2 (M-ext, funct7=0x01) |
| `0010011` | I | **ADDI** rd, rs1, imm | rd = rs1 + imm |
| `0000011` | I | **LW** rd, imm(rs1) | rd = mem[rs1 + imm] |
| `0100011` | S | **SW** rs2, imm(rs1) | mem[rs1 + imm] = rs2 |
| `1100011` | B | **BEQ/BNE/BLT/BGE** | conditional branches |
| `1101111` | J | **JAL** rd, off | rd = PC+4; PC += off |
| `0110111` | U | **LUI** rd, imm | rd = imm << 12 |
| `0001111` | — | **FENCE** | Thread barrier (`__syncthreads`) |
| `1110011` | — | **ECALL** | Thread completion |

### Custom-0: Crossbar Matrix/Memory Extension (opcode `0x0B`)

The `custom-0` opcode is part of RISC-V's reserved custom space. This compiler uses `funct3` to dispatch four crossbar operations:

| funct3 | Mnemonic | Operation | Physical Mechanism |
|--------|----------|-----------|-------------------|
| `000` | **MVM** rd, rs1 | Analog 16×16 matrix-vector multiply. Reads input from `mem[rs1]`, writes result to `mem[rd]`. ≈1 cycle. | Kirchhoff's current law: I_Q = Σ_P V_P·G_{P,Q} |
| `001` | **SLDR** rd, rs1 | Scratchpad load: `rd = scratch[rs1]` | Low-latency SRAM-like access |
| `010` | **SSTR** rs1, rs2 | Scratchpad store: `scratch[rs1] = rs2` | Low-latency SRAM-like access |
| `011` | **CSET** rd, rs1, rs2 | Program crossbar cell: `crossbar[rs1][rs2] = conductance(rd)` | Pot/dep pulse sequence to target conductance |

**Binary encoding (R-type, custom-0):**
```
  31      25 24   20 19   15 14  12 11    7 6      0
 ┌──────────┬───────┬───────┬──────┬───────┬────────┐
 │ funct7   │  rs2  │  rs1  │funct3│  rd   │0001011 │
 └──────────┴───────┴───────┴──────┴───────┴────────┘
```

### Custom-1: Crossbar Integer Arithmetic Extension (opcode `0x2B`)

The `custom-1` opcode (`0101011` in binary) is unused in standard RISC-V and allocated here for **crossbar-native integer arithmetic**. These instructions route operands through dedicated memristor scratch cells (column 62, rows 32–40) and exploit the physics of the device to perform the computation in the analog domain.

| funct3 | Mnemonic | Operation | Physical Mechanism | Scratch Rows (col 62) |
|--------|----------|-----------|-------------------|-----------------------|
| `000` | **XADD** rd, rs1, rs2 | rd = rs1 + rs2 | Conductance superposition: G_R = G(A) + G(B) − G_MIN | A=32, B=33, R=34 |
| `001` | **XSUB** rd, rs1, rs2 | rd = rs1 − rs2 | Differential pair: G_R = G(A) − G(B) + G_MIN | A=35, B=36, R=37 |
| `010` | **XMUL** rd, rs1, rs2 | rd = rs1 × rs2 | Ohm's law: I = V(A) · G(B), then sense current | V=38, G=39, R=40 |

**Binary encoding (R-type, custom-1):**
```
  31      25 24   20 19   15 14  12 11    7 6      0
 ┌──────────┬───────┬───────┬──────┬───────┬────────┐
 │ 0000000  │  rs2  │  rs1  │funct3│  rd   │0101011 │
 └──────────┴───────┴───────┴──────┴───────┴────────┘
```

**Physical detail for each operation:**

```
XADD — Conductance superposition at bit-line
  Step 1: G[32][62] = G_MIN + A/255 × (G_MAX − G_MIN)   ← program cell A
  Step 2: G[33][62] = G_MIN + B/255 × (G_MAX − G_MIN)   ← program cell B
  Step 3: G_R = G[32][62] + G[33][62] − G_MIN            ← bit-line reads both
       →  result = (G_R − G_MIN) / (G_MAX − G_MIN) × 255 = A + B  ✓

XSUB — Differential conductance pair
  Step 1: G[35][62] = G(A)     ← positive cell
  Step 2: G[36][62] = G(B)     ← negative cell
  Step 3: G_R = G(A) − G(B) + G_MIN  ← differential read
       →  result = A − B  (clamps to 0 for unsigned underflow)

XMUL — Ohm's law: I = V × G
  Step 1: G[38][62] = G(A)     ← voltage-control cell (V = A/255 × V_READ)
  Step 2: G[39][62] = G(B)     ← conductance-weight cell
  Step 3: I_µA = (A/255 × V_READ) × G(B)   ← current sense
       →  result = round(I / I_MAX × 255²)  (analog output shown in visualizer)
       →  rd = A × B  (exact RISC-V integer semantics preserved)
```

### Register File (x0–x31, RISC-V ABI)

| Registers | ABI Name | Role |
|-----------|----------|------|
| x0 | `zero` | Hardwired 0 |
| x5–x7 | `t0–t2` | Allocatable temporaries |
| x8–x9 | `s0–s1` | Allocatable saved registers |
| x10–x17 | `a0–a7` | Allocatable argument/result registers |
| x18–x24 | `s2–s8` | Allocatable saved registers |
| x25 | `tid` | **threadIdx** — thread index within block |
| x26 | `bid` | **blockIdx** — current block index |
| x27 | `bdm` | **blockDim** — threads per block |
| x28–x31 | `t3–t6` | Reserved |

Linear scan allocates **20 GPRs** (x5–x24). Every write to x0–x31 **also programs crossbar col 0, row = register number** via pot/dep pulses.

---

## Crossbar Physical Layout (64×64)

```
         col 0      col 1 … col 16    col 17 … col 61   col 62     col 63
         (REG FILE) (MVM WEIGHTS  )   (unused       )   (XA SCRATCH) …

row 0    x0 reg     W[0][1]…W[0][16]  —                 —           —
row 1    x1 reg     W[1][1]…W[1][16]  —                 —           —
…        …          …                 —                 —           —
row 31   x31 reg    W[31][1]          —                 —           —
row 32   —          —                 —                 XADD.A      —
row 33   —          —                 —                 XADD.B      —
row 34   —          —                 —                 XADD.R      —
row 35   —          —                 —                 XSUB.A      —
row 36   —          —                 —                 XSUB.B      —
row 37   —          —                 —                 XSUB.R      —
row 38   —          —                 —                 XMUL.V      —
row 39   —          —                 —                 XMUL.G      —
row 40   —          —                 —                 XMUL.R      —
…        —          —                 —                 —           —
row 63   —          —                 —                 —           —
```

---

## Performance Comparison

| Operation | Standard RISC-V (digital) | Crossbar IMC (analog) | Speedup |
|-----------|--------------------------|----------------------|---------|
| 16×16 MatVec (MVM) | 256 MUL + 255 ADD ≈ 511 ops, ~2560 cycles | 1 MVM ≈ **1 cycle** | **~2560×** |
| Integer ADD (XADD) | 1 ADD, 5 pipeline cycles | 1 XADD ≈ **2 cycles** | ~2.5× |
| Integer SUB (XSUB) | 1 SUB, 5 pipeline cycles | 1 XSUB ≈ **2 cycles** | ~2.5× |
| Integer MUL (XMUL) | 1 MUL, ~10 cycles (multi-cycle) | 1 XMUL ≈ **3 cycles** | ~3.3× |
| Register write | Register file write | Reg write + crossbar pot/dep | +∆t for crossbar |

The main advantage of XADD/XSUB/XMUL over standard ALU ops is **energy**: the analog computation avoids full-swing digital switching across all bits, and the result is available from a single current sense.

---

## The DSL

A minimal C-like language extended with crossbar instructions:

### Keywords

| Keyword | Type | Syntax | Compiles To |
|---------|------|--------|-------------|
| `kernel` | declaration | `kernel name(params) { }` | Function entry |
| `global` | param qualifier | `global int* ptr` | Pointer to 64-byte memory region |
| `shared` | storage | `shared int arr[N]` | Scratchpad allocation |
| `__syncthreads()` | barrier | `__syncthreads();` | `FENCE` instruction |
| `mvm()` | crossbar MVM | `mvm(out, in)` | `MVM` (custom-0, funct3=0) |
| `cset()` | crossbar write | `cset(row, col, val)` | `CSET` (custom-0, funct3=3) |
| `xadd()` | crossbar add | `xadd(dest, a, b)` | `XADD` (custom-1, funct3=0) |
| `xsub()` | crossbar sub | `xsub(dest, a, b)` | `XSUB` (custom-1, funct3=1) |
| `xmul()` | crossbar mul | `xmul(dest, a, b)` | `XMUL` (custom-1, funct3=2) |
| `threadIdx` | builtin | expression | `x25` register |
| `blockIdx` | builtin | expression | `x26` register |
| `blockDim` | builtin | expression | `x27` register |

### Examples

```c
// Standard parallel kernel
kernel vector_add(global int* a, global int* b, global int* c) {
    int i = blockIdx * blockDim + threadIdx;
    c[i] = a[i] + b[i];
}
```

```c
// Crossbar MVM: each thread programs a weight row, then runs analog matmul
kernel crossbar_mvm(global int* weights, global int* input, global int* output) {
    int tid = threadIdx;
    int w = weights[tid];
    cset(tid, 1, w);       // program crossbar[tid][1] = G(w) via pot/dep pulses
    __syncthreads();
    if (tid < 1) {
        mvm(output, input); // I_Q = Σ_P V_P · G_{P,Q} in single analog step
    }
}
```

```c
// Crossbar arithmetic: XADD / XSUB / XMUL via custom-1 opcode
kernel crossbar_arith(global int* a, global int* b, global int* out) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    int s = 0;
    int d = 0;
    int p = 0;
    xadd(s, ai, bi);   // XADD: conductance superposition  G(A)+G(B)−G_MIN
    xsub(d, ai, bi);   // XSUB: differential pair          G(A)−G(B)+G_MIN
    xmul(p, ai, bi);   // XMUL: Ohm's law                  I = V(A)·G(B)
    out[idx] = s;
}
```

```c
// Shared memory with barrier synchronization
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

---

## Visualizer Panels

### Left: Source Code Editor
Live-editable `.tgc` kernel source. Recompiles with 300ms debounce.

### Center: Compilation Pipeline
Three IR stages shown side-by-side:
1. **RISC-V IMC Dialect** — SSA form with `riscv.xadd`, `riscv.cset`, `riscv.mvm` etc.
2. **Optimization Passes** — constant propagation, register reuse, dead-code elimination
3. **Register Allocation** — physical x5–x24 assignments

Below: **32-bit Binary View** — color-coded by field (opcode red, rd blue, funct3 yellow, rs1 green, imm/rs2 orange).

### Right: IMC Execution + Analysis

**IMC Execution tab:**
- Thread cards showing pipeline stage, PC, and register values
- **Register File Crossbar Panel** (col 0) — all 32 registers with conductance in µS
- **Weight Matrix Heatmap** (cols 1–8, rows 0–7) — heat-map of programmed MVM weights
- **Crossbar Arithmetic Panel** — XADD/XSUB/XMUL scratch cells (col 62) with conductance bars and physics formulas, activated when custom-1 instructions execute
- **Memristor Write Log** — recent pot/dep pulse events: cell address, ΔG, pulse count

**Analysis tab:**
- IMC Performance Score (memory efficiency, branch uniformity, register pressure, crossbar utilization)
- Metrics grid: instructions, registers, compute/memory ops, MVM/CSET/XADD+XSUB+XMUL counts
- Thread divergence analysis
- Memory access coalescence patterns
- Workload balance (IMC vs Compute vs Memory)
- Estimated cycle count

---

## Built With

- **TypeScript + React** — all compiler and simulator logic runs in-browser
- **Vite** — build tooling
- **MLIR dialect naming conventions** — IR stages follow `riscv-imc.func` notation
- **RISC-V specification** — RV32IM base + custom-0/custom-1 extensions
- **Physical model** — Nature 633, Sep 2024 (molecular memristor device parameters)

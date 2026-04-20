import { CompilationTrace } from '../compiler/types';
import { compileTGC } from '../compiler/TGCCompiler';

export interface Example {
  name: string;
  description: string;
  source: string;
  trace: CompilationTrace;
  initialMemory: number[];
  numBlocks: number;
  threadsPerBlock: number;
}

const vectorAddSource = `kernel vector_add(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    c[idx] = a[idx] + b[idx];
}`;

const matrixMultiplySource = `kernel matrix_multiply(global int* A, global int* B, global int* C, int N) {
    int idx = blockIdx * blockDim + threadIdx;
    int row = idx / N;
    int col = idx - row * N;
    int sum = 0;
    for (int k = 0; k < N; k = k + 1) {
        int a_val = A[row * N + k];
        int b_val = B[k * N + col];
        sum = sum + a_val * b_val;
    }
    C[idx] = sum;
}`;

const dotProductSource = `kernel dot_product(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    c[idx] = ai * bi;
}`;

const reluSource = `kernel relu(global int* input, global int* output) {
    int idx = blockIdx * blockDim + threadIdx;
    int val = input[idx];
    if (val > 0) {
        output[idx] = val;
    } else {
        output[idx] = 0;
    }
}`;

const saxpySource = `kernel saxpy(global int* x, global int* y, int a) {
    int idx = blockIdx * blockDim + threadIdx;
    int xi = x[idx];
    int yi = y[idx];
    y[idx] = a * xi + yi;
}`;

const conv1dSource = `kernel conv1d(global int* input, global int* weights, global int* output, int K) {
    int idx = blockIdx * blockDim + threadIdx;
    int sum = 0;
    for (int j = 0; j < K; j = j + 1) {
        int in_val = input[idx + j];
        int w_val = weights[j];
        sum = sum + in_val * w_val;
    }
    output[idx] = sum;
}`;

const vectorMaxSource = `kernel vector_max(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    if (ai > bi) {
        c[idx] = ai;
    } else {
        c[idx] = bi;
    }
}`;

// Crossbar variable load/store: custom-2 (0x5B) CVLD/CVST with safe memory mapping.
// Variables are stored directly in the memristor data region (cols 17-61), bypassing
// global DRAM — then consumed by custom-1 crossbar arithmetic. Reads use V ≤ 500mV
// (non-disturbing); writes are bounds-checked so reserved regions are never corrupted.
const crossbarVarSource = `kernel crossbar_var(global int* a, global int* b, global int* out) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    // Store operands directly into the memristor data region (cols 17, 18)
    cvst(ai, threadIdx, 17);
    cvst(bi, threadIdx, 18);
    // Load them back non-destructively (V_read <= 500mV, below V_th=830mV)
    int va = 0;
    int vb = 0;
    cvld(va, threadIdx, 17);
    cvld(vb, threadIdx, 18);
    // Crossbar-native addition on the reloaded operands
    int s = 0;
    xadd(s, va, vb);
    out[idx] = s;
}`;

// Crossbar arithmetic: XADD + XSUB + XMUL using custom-1 (0x2B) opcode
const crossbarArithSource = `kernel crossbar_arith(global int* a, global int* b, global int* out) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    int s = 0;
    int d = 0;
    int p = 0;
    xadd(s, ai, bi);
    xsub(d, ai, bi);
    xmul(p, ai, bi);
    out[idx] = s;
}`;

// Crossbar MVM: each thread programs one row of the weight matrix, then runs analog MVM
const crossbarMVMSource = `kernel crossbar_mvm(global int* weights, global int* input, global int* output) {
    // Phase 1: each thread loads its weight and programs one crossbar row
    int tid = threadIdx;
    int w = weights[tid];
    cset(tid, 1, w);
    __syncthreads();
    // Phase 2: thread 0 runs the analog matrix-vector multiply
    if (tid < 1) {
        mvm(output, input);
    }
}`;

// NEW: Shared memory tiled vector add - demonstrates shared mem + syncthreads
const sharedTileAddSource = `kernel shared_tile_add(global int* a, global int* b, global int* c) {
    shared int tile_a[4];
    shared int tile_b[4];
    int idx = blockIdx * blockDim + threadIdx;
    tile_a[threadIdx] = a[idx];
    tile_b[threadIdx] = b[idx];
    __syncthreads();
    int sum = tile_a[threadIdx] + tile_b[threadIdx];
    c[idx] = sum;
}`;

// NEW: Shared memory reduction - demonstrates shared mem cooperation
const sharedReductionSource = `kernel shared_reduce(global int* input, global int* output) {
    shared int scratch[4];
    int idx = blockIdx * blockDim + threadIdx;
    scratch[threadIdx] = input[idx];
    __syncthreads();
    if (threadIdx < 2) {
        scratch[threadIdx] = scratch[threadIdx] + scratch[threadIdx + 2];
    }
    __syncthreads();
    if (threadIdx < 1) {
        scratch[0] = scratch[0] + scratch[1];
        output[blockIdx] = scratch[0];
    }
}`;

export const EXAMPLES: Example[] = [
  {
    name: 'Crossbar Variables',
    description: 'CVLD + CVST (custom-2) → memristor-backed variables',
    source: crossbarVarSource,
    trace: compileTGC(crossbarVarSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      mem[0] = 30; mem[1] = 20; mem[2] = 50; mem[3] = 40;
      mem[64] = 10; mem[65] = 5; mem[66] = 20; mem[67] = 15;
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: 'Crossbar Arith',
    description: 'XADD + XSUB + XMUL (custom-1)',
    source: crossbarArithSource,
    trace: compileTGC(crossbarArithSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      // a[0..3] = [30, 20, 50, 40]
      mem[0] = 30; mem[1] = 20; mem[2] = 50; mem[3] = 40;
      // b[0..3] = [10, 5,  20, 15]
      mem[64] = 10; mem[65] = 5; mem[66] = 20; mem[67] = 15;
      // expected sum: [40,25,70,55], diff: [20,15,30,25], mul: [300,100,1000,600]
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: 'Crossbar MVM',
    description: 'CSET weights → analog MVM',
    source: crossbarMVMSource,
    trace: compileTGC(crossbarMVMSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      // weights[0..3] at mem[0..3]: high conductances for visible MVM result
      mem[0] = 255; mem[1] = 200; mem[2] = 150; mem[3] = 100;
      // input[0..3] at mem[64..67]
      mem[64] = 100; mem[65] = 150; mem[66] = 200; mem[67] = 250;
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: 'Vector Add',
    description: 'c[i] = a[i] + b[i]',
    source: vectorAddSource,
    trace: compileTGC(vectorAddSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Shared Tile Add',
    description: 'Tiled add with shared memory',
    source: sharedTileAddSource,
    trace: compileTGC(sharedTileAddSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Shared Reduce',
    description: 'Parallel reduction with shared memory',
    source: sharedReductionSource,
    trace: compileTGC(sharedReductionSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      // input = [1, 2, 3, 4, 5, 6, 7, 8]
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Matrix Multiply',
    description: 'C = A * B (2x2)',
    source: matrixMultiplySource,
    trace: compileTGC(matrixMultiplySource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      mem[0] = 1; mem[1] = 2; mem[2] = 3; mem[3] = 4;
      mem[64] = 5; mem[65] = 6; mem[66] = 7; mem[67] = 8;
      mem[192] = 2;
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: '1D Convolution',
    description: 'output[i] = sum(input[i+j] * w[j])',
    source: conv1dSource,
    trace: compileTGC(conv1dSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      mem[64] = 1; mem[65] = 2; mem[66] = 1;
      mem[192] = 3;
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: 'Dot Product',
    description: 'c[i] = a[i] * b[i]',
    source: dotProductSource,
    trace: compileTGC(dotProductSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      const a = [2, 3, 4, 5, 1, 2, 3, 4];
      for (let i = 0; i < a.length; i++) mem[i] = a[i];
      const b = [5, 4, 3, 2, 6, 7, 8, 9];
      for (let i = 0; i < b.length; i++) mem[64 + i] = b[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'ReLU Activation',
    description: 'output[i] = max(0, input[i])',
    source: reluSource,
    trace: compileTGC(reluSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      const input = [0, 5, 0, 12, 0, 3, 0, 7];
      for (let i = 0; i < input.length; i++) mem[i] = input[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'SAXPY',
    description: 'y[i] = a * x[i] + y[i]',
    source: saxpySource,
    trace: compileTGC(saxpySource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      mem[192] = 3;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Vector Max',
    description: 'c[i] = max(a[i], b[i])',
    source: vectorMaxSource,
    trace: compileTGC(vectorMaxSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      const a = [3, 7, 2, 9, 1, 8, 4, 6];
      for (let i = 0; i < a.length; i++) mem[i] = a[i];
      const b = [5, 4, 6, 1, 8, 3, 7, 2];
      for (let i = 0; i < b.length; i++) mem[64 + i] = b[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
];

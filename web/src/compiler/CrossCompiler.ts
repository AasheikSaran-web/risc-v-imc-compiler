/**
 * CrossCompiler.ts
 * C / Python → RISC-V + Memristor Crossbar Assembly compiler.
 *
 * Produces 6 compilation steps for the visualizer plus a final assembly string.
 * All errors are caught and placed in the `errors` array — this function never throws.
 */

import { mapVariables, VariableInfo, MappingResult, VarAllocation } from './MemoryMapper';

export type Lang = 'c' | 'python';

export interface CompileStep {
  title: string;
  content: string;
}

export interface CrossCompileResult {
  steps: CompileStep[];
  assembly: string;
  mapping: MappingResult;
  errors: string[];
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

type TokKind =
  | 'keyword' | 'ident' | 'number' | 'string'
  | 'op' | 'punct' | 'newline' | 'indent' | 'dedent' | 'eof' | 'unknown';

interface Tok {
  kind: TokKind;
  value: string;
  line: number;
}

const C_KEYWORDS   = new Set(['int','float','void','return','for','while','if','else','break','continue','def','struct','typedef','const','static','unsigned']);
const PY_KEYWORDS  = new Set(['def','return','for','while','if','else','elif','break','continue','in','range','True','False','None','and','or','not','import','from','lambda','pass']);
const OPS          = ['==','!=','<=','>=','+=','-=','*=','/=','->','**','//','<<','>>','&&','||','++','--','+=','-='];
const SINGLE_CHARS = new Set('+-*/%=<>!&|^~()[]{},.:;@#');

function tokenize(source: string, _lang: Lang): Tok[] {
  const tokens: Tok[] = [];
  const lines = source.split('\n');

  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    const lineNum = li + 1;
    let col = 0;

    while (col < line.length) {
      // Skip whitespace (not newlines — those are handled per-line)
      if (line[col] === ' ' || line[col] === '\t') { col++; continue; }
      // Comment
      if (line[col] === '#' || (line[col] === '/' && line[col+1] === '/')) break;
      if (line[col] === '/' && line[col+1] === '*') { col += 2; while (col < line.length && !(line[col-1]==='*' && line[col]==='/')) col++; col++; continue; }

      // Number
      if (/\d/.test(line[col]) || (line[col] === '-' && col+1 < line.length && /\d/.test(line[col+1]) && tokens.length > 0 && ['op','punct'].includes(tokens[tokens.length-1].kind))) {
        let num = '';
        if (line[col] === '-') { num = '-'; col++; }
        while (col < line.length && /[\d._xXa-fA-F]/.test(line[col])) { num += line[col]; col++; }
        tokens.push({ kind: 'number', value: num, line: lineNum });
        continue;
      }

      // String literal
      if (line[col] === '"' || line[col] === "'") {
        const q = line[col]; col++;
        let s = '';
        while (col < line.length && line[col] !== q) { s += line[col]; col++; }
        col++;
        tokens.push({ kind: 'string', value: s, line: lineNum });
        continue;
      }

      // Identifier / keyword
      if (/[a-zA-Z_]/.test(line[col])) {
        let id = '';
        while (col < line.length && /[a-zA-Z0-9_]/.test(line[col])) { id += line[col]; col++; }
        const kw = C_KEYWORDS.has(id) || PY_KEYWORDS.has(id);
        tokens.push({ kind: kw ? 'keyword' : 'ident', value: id, line: lineNum });
        continue;
      }

      // Two-char operators
      const two = line.slice(col, col+2);
      if (OPS.includes(two)) { tokens.push({ kind: 'op', value: two, line: lineNum }); col += 2; continue; }

      // Single-char punctuation/operators
      if (SINGLE_CHARS.has(line[col])) {
        const k: TokKind = /[+\-*/%=<>!&|^~]/.test(line[col]) ? 'op' : 'punct';
        tokens.push({ kind: k, value: line[col], line: lineNum });
        col++;
        continue;
      }

      tokens.push({ kind: 'unknown', value: line[col], line: lineNum });
      col++;
    }
  }
  tokens.push({ kind: 'eof', value: '', line: lines.length });
  return tokens;
}

// ── Simple AST types ─────────────────────────────────────────────────────────

interface ASTParam { name: string; type: string; isPtr: boolean; }
interface ASTFunc {
  name: string;
  params: ASTParam[];
  body: ASTStmt[];
  returnType: string;
}

type ASTStmt =
  | { kind: 'decl';   vtype: string; name: string; size?: number; init?: ASTExpr }
  | { kind: 'assign'; target: string; index?: ASTExpr; value: ASTExpr }
  | { kind: 'for';    init: ASTStmt; cond: ASTExpr; step: ASTStmt; body: ASTStmt[] }
  | { kind: 'foreach';varName: string; iterable: ASTExpr; body: ASTStmt[] }
  | { kind: 'if';     cond: ASTExpr; then: ASTStmt[]; els: ASTStmt[] }
  | { kind: 'return'; value?: ASTExpr }
  | { kind: 'expr';   expr: ASTExpr }
  | { kind: 'unsupported'; text: string };

type ASTExpr =
  | { ek: 'num';    value: number }
  | { ek: 'ident';  name: string }
  | { ek: 'index';  array: string; idx: ASTExpr }
  | { ek: 'call';   func: string; args: ASTExpr[] }
  | { ek: 'binop';  op: string; lhs: ASTExpr; rhs: ASTExpr }
  | { ek: 'unop';   op: string; expr: ASTExpr }
  | { ek: 'array_lit'; elements: ASTExpr[] };

// ── Parser ───────────────────────────────────────────────────────────────────

function parse(tokens: Tok[], lang: Lang): ASTFunc[] {
  let pos = 0;
  const peek  = () => tokens[pos] ?? { kind: 'eof', value: '', line: 0 };
  const eat   = () => tokens[pos++] ?? { kind: 'eof', value: '', line: 0 };
  const at    = (v: string) => peek().value === v;
  const atK   = (k: TokKind) => peek().kind === k;
  const expect = (v: string) => { if (peek().value === v) eat(); };

  function parseExpr(): ASTExpr {
    return parseBinOp(0);
  }

  const PREC: Record<string, number> = {
    '||':1,'&&':2,'|':3,'^':4,'&':5,
    '==':6,'!=':6,'<':7,'<=':7,'>':7,'>=':7,
    '<<':8,'>>':8,'+':9,'-':9,'*':10,'/':10,'%':10,'**':11,'//':10
  };

  function parseBinOp(minPrec: number): ASTExpr {
    let lhs = parseUnary();
    while (true) {
      const op = peek().value;
      const pr = PREC[op];
      if (pr === undefined || pr <= minPrec) break;
      eat();
      const rhs = parseBinOp(pr);
      lhs = { ek: 'binop', op, lhs, rhs };
    }
    return lhs;
  }

  function parseUnary(): ASTExpr {
    if (at('-') || at('!') || at('~') || at('*') || at('&')) {
      const op = eat().value;
      return { ek: 'unop', op, expr: parseUnary() };
    }
    return parsePrimary();
  }

  function parsePrimary(): ASTExpr {
    // Number
    if (atK('number')) {
      return { ek: 'num', value: parseFloat(eat().value) };
    }
    // Paren group
    if (at('(')) {
      eat(); const e = parseExpr(); expect(')'); return e;
    }
    // Array literal (Python)
    if (at('[')) {
      eat();
      const elems: ASTExpr[] = [];
      while (!at(']') && !atK('eof')) {
        elems.push(parseExpr());
        if (at(',')) eat();
      }
      expect(']');
      return { ek: 'array_lit', elements: elems };
    }
    // Identifier, call, or index
    if (atK('ident') || atK('keyword')) {
      const name = eat().value;
      if (at('(')) {
        eat();
        const args: ASTExpr[] = [];
        while (!at(')') && !atK('eof')) {
          args.push(parseExpr());
          if (at(',')) eat();
        }
        expect(')');
        return { ek: 'call', func: name, args };
      }
      if (at('[')) {
        eat();
        const idx = parseExpr();
        expect(']');
        // Could be double-indexed; just return single level
        return { ek: 'index', array: name, idx };
      }
      return { ek: 'ident', name };
    }
    // Fallback
    const tok = eat();
    return { ek: 'num', value: 0 };
  }

  // ── Statement parsing ──────────────────────────────────────────────────────

  function parseBlock(stopAt?: string): ASTStmt[] {
    const stmts: ASTStmt[] = [];
    if (lang === 'c') {
      expect('{');
      while (!at('}') && !atK('eof')) stmts.push(parseCStmt());
      expect('}');
    } else {
      // Python: collect indented lines (simplified — no real indent tracking)
      expect(':');
      while (!atK('eof') && peek().line > (stmts[0] ? 0 : 0)) {
        const cur = peek();
        if (stopAt && cur.value === stopAt) break;
        const stmt = parsePyStmt();
        stmts.push(stmt);
        if (stmts.length > 100) break;
      }
    }
    return stmts;
  }

  function parseCStmt(): ASTStmt {
    // int x = ...; OR int arr[N];
    if (peek().value === 'int' || peek().value === 'float' || peek().value === 'unsigned') {
      const vtype = eat().value;
      const name  = eat().value;
      if (at('[')) {
        eat();
        let sz = 0;
        if (atK('number')) sz = parseInt(eat().value);
        expect(']');
        let init: ASTExpr | undefined;
        if (at('=')) { eat(); init = parseExpr(); }
        expect(';');
        return { kind: 'decl', vtype, name, size: sz };
      }
      let init: ASTExpr | undefined;
      if (at('=')) { eat(); init = parseExpr(); }
      expect(';');
      return { kind: 'decl', vtype, name, init };
    }
    // for (int i = 0; i < n; i++)
    if (peek().value === 'for') {
      eat(); expect('(');
      const initStmt = parseCStmt();
      const cond = parseExpr(); expect(';');
      // step: i++ or i += 1 or i = i + 1
      let stepStmt: ASTStmt;
      const stepName = peek().value; eat();
      if (at('++')) { eat(); stepStmt = { kind: 'assign', target: stepName, value: { ek: 'binop', op: '+', lhs: { ek: 'ident', name: stepName }, rhs: { ek: 'num', value: 1 } } }; }
      else if (at('+=')) { eat(); const v = parseExpr(); stepStmt = { kind: 'assign', target: stepName, value: { ek: 'binop', op: '+', lhs: { ek: 'ident', name: stepName }, rhs: v } }; }
      else { expect('='); const v = parseExpr(); stepStmt = { kind: 'assign', target: stepName, value: v }; }
      expect(')');
      const body = parseBlock();
      return { kind: 'for', init: initStmt, cond, step: stepStmt, body };
    }
    // if/else
    if (peek().value === 'if') {
      eat(); expect('(');
      const cond = parseExpr(); expect(')');
      const then = parseBlock();
      let els: ASTStmt[] = [];
      if (peek().value === 'else') { eat(); els = parseBlock(); }
      return { kind: 'if', cond, then, els };
    }
    // return
    if (peek().value === 'return') {
      eat();
      if (at(';')) { eat(); return { kind: 'return' }; }
      const v = parseExpr(); expect(';');
      return { kind: 'return', value: v };
    }
    // assignment or function call
    if (atK('ident')) {
      const name = eat().value;
      if (at('[')) {
        eat(); const idx = parseExpr(); expect(']'); expect('=');
        const val = parseExpr(); expect(';');
        return { kind: 'assign', target: name, index: idx, value: val };
      }
      if (at('=')) { eat(); const val = parseExpr(); expect(';'); return { kind: 'assign', target: name, value: val }; }
      if (at('+=')) { eat(); const val = parseExpr(); expect(';'); return { kind: 'assign', target: name, value: { ek: 'binop', op: '+', lhs: { ek: 'ident', name }, rhs: val } }; }
      if (at('-=')) { eat(); const val = parseExpr(); expect(';'); return { kind: 'assign', target: name, value: { ek: 'binop', op: '-', lhs: { ek: 'ident', name }, rhs: val } }; }
      if (at('(')) {
        eat(); const args: ASTExpr[] = [];
        while (!at(')') && !atK('eof')) { args.push(parseExpr()); if (at(',')) eat(); }
        expect(')'); expect(';');
        return { kind: 'expr', expr: { ek: 'call', func: name, args } };
      }
      expect(';');
      return { kind: 'expr', expr: { ek: 'ident', name } };
    }
    // Skip unknown
    const tok = eat();
    return { kind: 'unsupported', text: tok.value };
  }

  function parsePyStmt(): ASTStmt {
    const cur = peek();
    // def — handled at top level
    if (cur.value === 'def') return { kind: 'unsupported', text: 'nested-def' };
    // for i in range(n):
    if (cur.value === 'for') {
      eat();
      const varName = eat().value;
      // "in"
      eat();
      const iterExpr = parseExpr();
      expect(':');
      // Collect body lines (those with higher line number)
      const startLine = peek().line;
      const body: ASTStmt[] = [];
      while (!atK('eof') && peek().line > cur.line) {
        body.push(parsePyStmt());
        if (body.length > 50) break;
      }
      return { kind: 'foreach', varName, iterable: iterExpr, body };
    }
    // if cond:
    if (cur.value === 'if') {
      eat();
      const cond = parseExpr();
      expect(':');
      const then: ASTStmt[] = [];
      const startLine = peek().line;
      while (!atK('eof') && peek().line > cur.line && peek().value !== 'else' && peek().value !== 'elif') {
        then.push(parsePyStmt());
        if (then.length > 50) break;
      }
      const els: ASTStmt[] = [];
      if (peek().value === 'else') {
        eat(); expect(':');
        while (!atK('eof') && peek().line > cur.line) {
          els.push(parsePyStmt());
          if (els.length > 50) break;
        }
      }
      return { kind: 'if', cond, then, els };
    }
    // return
    if (cur.value === 'return') {
      eat();
      if (atK('eof') || peek().line > cur.line) return { kind: 'return' };
      const v = parseExpr();
      return { kind: 'return', value: v };
    }
    // x = expr  or  arr[i] = expr
    if (atK('ident')) {
      const name = eat().value;
      if (at('[')) {
        eat(); const idx = parseExpr(); expect(']'); expect('=');
        const val = parseExpr();
        return { kind: 'assign', target: name, index: idx, value: val };
      }
      if (at('=') && peek().value !== '==') {
        eat(); const val = parseExpr();
        // Infer array decl for "arr = [0] * N" or "arr = [1,2,3]"
        if (val.ek === 'binop' && val.op === '*' && val.lhs.ek === 'array_lit') {
          const sz = val.rhs.ek === 'num' ? val.rhs.value : 8;
          return { kind: 'decl', vtype: 'int', name, size: sz };
        }
        if (val.ek === 'array_lit') {
          return { kind: 'decl', vtype: 'int', name, size: val.elements.length };
        }
        // Check if this is a first-use (simple heuristic)
        return { kind: 'assign', target: name, value: val };
      }
      if (at('+=')) { eat(); const val = parseExpr(); return { kind: 'assign', target: name, value: { ek: 'binop', op: '+', lhs: { ek: 'ident', name }, rhs: val } }; }
      if (at('(')) {
        eat(); const args: ASTExpr[] = [];
        while (!at(')') && !atK('eof')) { args.push(parseExpr()); if (at(',')) eat(); }
        expect(')');
        return { kind: 'expr', expr: { ek: 'call', func: name, args } };
      }
      return { kind: 'expr', expr: { ek: 'ident', name } };
    }
    const tok = eat();
    return { kind: 'unsupported', text: tok.value };
  }

  // ── Top-level function parsing ─────────────────────────────────────────────

  const funcs: ASTFunc[] = [];

  while (!atK('eof')) {
    // C: int/void funcname(params) { body }
    // Python: def funcname(params):
    if (lang === 'c') {
      if (peek().value === 'int' || peek().value === 'void' || peek().value === 'float') {
        const retType = eat().value;
        if (!atK('ident')) { eat(); continue; }
        const fname = eat().value;
        expect('(');
        const params: ASTParam[] = [];
        while (!at(')') && !atK('eof')) {
          let ptype = '';
          let isPtr = false;
          if (atK('keyword') || atK('ident')) ptype = eat().value;
          if (atK('keyword') || atK('ident')) ptype += ' ' + eat().value;
          if (at('*')) { eat(); isPtr = true; }
          const pname = atK('ident') ? eat().value : '';
          params.push({ name: pname, type: ptype, isPtr });
          if (at(',')) eat();
          if (at('[')) { eat(); expect(']'); }
        }
        expect(')');
        const body: ASTStmt[] = [];
        if (at('{')) {
          eat();
          while (!at('}') && !atK('eof')) body.push(parseCStmt());
          expect('}');
        }
        funcs.push({ name: fname, params, body, returnType: retType });
      } else {
        eat(); // skip unknown top-level token
      }
    } else {
      // Python
      if (peek().value === 'def') {
        eat();
        const fname = eat().value;
        expect('(');
        const params: ASTParam[] = [];
        while (!at(')') && !atK('eof')) {
          const pname = eat().value;
          let isPtr = false;
          let ptype = 'int';
          // type annotations: name: type
          if (at(':')) { eat(); ptype = eat().value; }
          // pointer hint for list params
          if (ptype === 'list' || pname.endsWith('s') || pname.endsWith('arr') || pname.endsWith('out')) isPtr = true;
          params.push({ name: pname, type: ptype, isPtr });
          if (at(',')) eat();
        }
        expect(')');
        if (at('->')) { eat(); eat(); } // return type annotation
        expect(':');
        const curLine = peek().line;
        const body: ASTStmt[] = [];
        while (!atK('eof') && (peek().kind !== 'keyword' || peek().value !== 'def')) {
          body.push(parsePyStmt());
          if (body.length > 200) break;
        }
        funcs.push({ name: fname, params, body, returnType: 'int' });
      } else {
        eat();
      }
    }
  }

  // If no functions found, wrap the whole thing in a synthetic "main"
  if (funcs.length === 0) {
    pos = 0;
    const body: ASTStmt[] = [];
    while (!atK('eof')) {
      if (lang === 'c') body.push(parseCStmt());
      else body.push(parsePyStmt());
      if (body.length > 200) break;
    }
    funcs.push({ name: 'main', params: [], body, returnType: 'void' });
  }

  return funcs;
}

// ── Variable analysis ─────────────────────────────────────────────────────────

interface VarRecord {
  name: string;
  isArray: boolean;
  size: number;
  defLine: number;
  lastUseLine: number;
  accessCount: number;
}

function collectVarRefs(expr: ASTExpr, line: number, vars: Map<string, VarRecord>): void {
  switch (expr.ek) {
    case 'ident':
      if (vars.has(expr.name)) {
        const v = vars.get(expr.name)!;
        v.accessCount++;
        if (line > v.lastUseLine) v.lastUseLine = line;
      }
      break;
    case 'index':
      if (vars.has(expr.array)) {
        const v = vars.get(expr.array)!;
        v.accessCount++;
        if (line > v.lastUseLine) v.lastUseLine = line;
      }
      collectVarRefs(expr.idx, line, vars);
      break;
    case 'binop':
      collectVarRefs(expr.lhs, line, vars);
      collectVarRefs(expr.rhs, line, vars);
      break;
    case 'unop':
      collectVarRefs(expr.expr, line, vars);
      break;
    case 'call':
      expr.args.forEach(a => collectVarRefs(a, line, vars));
      break;
    case 'array_lit':
      expr.elements.forEach(e => collectVarRefs(e, line, vars));
      break;
  }
}

function analyzeStmts(stmts: ASTStmt[], vars: Map<string, VarRecord>, lineBase: number): void {
  stmts.forEach((stmt, si) => {
    const line = lineBase + si + 1;
    switch (stmt.kind) {
      case 'decl':
        vars.set(stmt.name, {
          name: stmt.name,
          isArray: stmt.size !== undefined && stmt.size > 0,
          size: stmt.size ?? 1,
          defLine: line,
          lastUseLine: line,
          accessCount: 0,
        });
        if (stmt.init) collectVarRefs(stmt.init, line, vars);
        break;
      case 'assign':
        if (vars.has(stmt.target)) {
          const v = vars.get(stmt.target)!;
          v.accessCount++;
          if (line > v.lastUseLine) v.lastUseLine = line;
        } else {
          // implicit decl for Python
          vars.set(stmt.target, { name: stmt.target, isArray: false, size: 1, defLine: line, lastUseLine: line, accessCount: 1 });
        }
        if (stmt.index) collectVarRefs(stmt.index, line, vars);
        collectVarRefs(stmt.value, line, vars);
        break;
      case 'for':
        analyzeStmts([stmt.init], vars, line);
        collectVarRefs(stmt.cond, line, vars);
        analyzeStmts(stmt.body, vars, line + 1);
        analyzeStmts([stmt.step], vars, line);
        break;
      case 'foreach':
        vars.set(stmt.varName, { name: stmt.varName, isArray: false, size: 1, defLine: line, lastUseLine: line, accessCount: 0 });
        collectVarRefs(stmt.iterable, line, vars);
        analyzeStmts(stmt.body, vars, line + 1);
        break;
      case 'if':
        collectVarRefs(stmt.cond, line, vars);
        analyzeStmts(stmt.then, vars, line + 1);
        analyzeStmts(stmt.els, vars, line + 1);
        break;
      case 'return':
        if (stmt.value) collectVarRefs(stmt.value, line, vars);
        break;
      case 'expr':
        collectVarRefs(stmt.expr, line, vars);
        break;
    }
  });
}

// ── IR generation ─────────────────────────────────────────────────────────────

function exprToStr(e: ASTExpr): string {
  switch (e.ek) {
    case 'num':   return String(e.value);
    case 'ident': return e.name;
    case 'index': return `${e.array}[${exprToStr(e.idx)}]`;
    case 'binop': return `${exprToStr(e.lhs)} ${e.op} ${exprToStr(e.rhs)}`;
    case 'unop':  return `${e.op}${exprToStr(e.expr)}`;
    case 'call':  return `${e.func}(${e.args.map(exprToStr).join(', ')})`;
    case 'array_lit': return `[${e.elements.map(exprToStr).join(', ')}]`;
  }
}

function generateIR(funcs: ASTFunc[], vars: Map<string, VarRecord>, allocs: VarAllocation[]): string[] {
  const lines: string[] = [];
  let tempIdx = 0;
  const nextTemp = () => `t${tempIdx++}`;

  // Build quick lookup: varname → allocation
  const allocMap = new Map<string, VarAllocation>();
  for (const a of allocs) allocMap.set(a.variable, a);

  function irExpr(e: ASTExpr): string {
    switch (e.ek) {
      case 'num':   return String(e.value);
      case 'ident': return e.name;
      case 'index': {
        const t = nextTemp();
        const alloc = allocMap.get(e.array);
        if (alloc && alloc.storage === 'crossbar' && alloc.cells && alloc.cells.length > 0) {
          const baseCol = alloc.cells[0].col;
          lines.push(`  ${t} = cvld crossbar[${exprToStr(e.idx)}][${baseCol}]   ; load ${e.array}[${exprToStr(e.idx)}]`);
        } else {
          lines.push(`  ${t} = ${e.array}[${exprToStr(e.idx)}]`);
        }
        return t;
      }
      case 'call': {
        if (e.func === 'range' && e.args.length === 1) return exprToStr(e.args[0]);
        const t = nextTemp();
        lines.push(`  ${t} = call ${e.func}(${e.args.map(exprToStr).join(', ')})`);
        return t;
      }
      case 'binop': {
        const l = irExpr(e.lhs);
        const r = irExpr(e.rhs);
        const t = nextTemp();
        const lAlloc = l.startsWith('t') ? undefined : allocMap.get(l);
        const rAlloc = r.startsWith('t') ? undefined : allocMap.get(r);
        if (lAlloc?.storage === 'crossbar' && rAlloc?.storage === 'crossbar') {
          const op = e.op === '+' ? 'xadd' : e.op === '-' ? 'xsub' : e.op === '*' ? 'xmul' : e.op;
          lines.push(`  ${t} = ${op} ${l}, ${r}   ; crossbar arithmetic`);
        } else {
          lines.push(`  ${t} = ${l} ${e.op} ${r}`);
        }
        return t;
      }
      case 'unop': {
        const t = nextTemp();
        lines.push(`  ${t} = ${e.op} ${irExpr(e.expr)}`);
        return t;
      }
      case 'array_lit': {
        const t = nextTemp();
        lines.push(`  ${t} = [${e.elements.map(exprToStr).join(', ')}]   ; array literal`);
        return t;
      }
    }
  }

  function irStmt(stmt: ASTStmt, depth: number = 0): void {
    const pad = '  '.repeat(depth);
    switch (stmt.kind) {
      case 'decl': {
        const alloc = allocMap.get(stmt.name);
        if (stmt.size && stmt.size > 0) {
          const col = alloc?.cells?.[0]?.col ?? 17;
          lines.push(`${pad}  ; array ${stmt.name}[${stmt.size}] → crossbar col ${col}`);
          for (let r = 0; r < Math.min(stmt.size, 4); r++) {
            lines.push(`${pad}  cvst 0, ${r}, ${col}   ; ${stmt.name}[${r}] = 0`);
          }
          if (stmt.size > 4) lines.push(`${pad}  ; ... (${stmt.size - 4} more cells initialized)`);
        } else {
          const reg = alloc?.register ?? 't0';
          const initStr = stmt.init ? exprToStr(stmt.init) : '0';
          lines.push(`${pad}  ${reg} = ${initStr}   ; ${stmt.name} → reg ${reg}`);
        }
        break;
      }
      case 'assign': {
        const alloc = allocMap.get(stmt.target);
        if (stmt.index) {
          const idxStr = irExpr(stmt.index);
          const valStr = irExpr(stmt.value);
          const col = alloc?.cells?.[0]?.col ?? 17;
          lines.push(`${pad}  cvst ${valStr}, ${idxStr}, ${col}   ; ${stmt.target}[${idxStr}] = ${valStr}`);
        } else {
          const valStr = irExpr(stmt.value);
          const reg = alloc?.register ?? stmt.target;
          lines.push(`${pad}  ${reg} = ${valStr}   ; ${stmt.target} = ${valStr}`);
        }
        break;
      }
      case 'for': {
        irStmt(stmt.init, depth);
        const condStr = exprToStr(stmt.cond);
        const lbl = `L${tempIdx++}`;
        lines.push(`${pad}${lbl}:`);
        lines.push(`${pad}  if !(${condStr}) goto exit_${lbl}`);
        stmt.body.forEach(s => irStmt(s, depth + 1));
        irStmt(stmt.step, depth);
        lines.push(`${pad}  goto ${lbl}`);
        lines.push(`${pad}exit_${lbl}:`);
        break;
      }
      case 'foreach': {
        const iterStr = exprToStr(stmt.iterable);
        const lbl = `L${tempIdx++}`;
        lines.push(`${pad}  ${stmt.varName} = 0`);
        lines.push(`${pad}${lbl}:`);
        lines.push(`${pad}  if ${stmt.varName} >= ${iterStr} goto exit_${lbl}`);
        stmt.body.forEach(s => irStmt(s, depth + 1));
        lines.push(`${pad}  ${stmt.varName} = ${stmt.varName} + 1`);
        lines.push(`${pad}  goto ${lbl}`);
        lines.push(`${pad}exit_${lbl}:`);
        break;
      }
      case 'if': {
        const condStr = exprToStr(stmt.cond);
        lines.push(`${pad}  if !(${condStr}) goto else_${tempIdx}`);
        const lbl = tempIdx++;
        stmt.then.forEach(s => irStmt(s, depth + 1));
        if (stmt.els.length > 0) {
          lines.push(`${pad}  goto end_${lbl}`);
          lines.push(`${pad}else_${lbl}:`);
          stmt.els.forEach(s => irStmt(s, depth + 1));
          lines.push(`${pad}end_${lbl}:`);
        } else {
          lines.push(`${pad}else_${lbl}:`);
        }
        break;
      }
      case 'return': {
        const val = stmt.value ? irExpr(stmt.value) : '0';
        lines.push(`${pad}  a0 = ${val}   ; return value`);
        lines.push(`${pad}  ret`);
        break;
      }
      case 'expr': {
        const s = irExpr(stmt.expr);
        lines.push(`${pad}  _ = ${s}`);
        break;
      }
      case 'unsupported':
        lines.push(`${pad}  ; (unsupported: ${stmt.text})`);
        break;
    }
  }

  for (const fn of funcs) {
    lines.push(`; function ${fn.name}(${fn.params.map(p => p.name).join(', ')})`);
    fn.body.forEach(s => irStmt(s));
    lines.push(`  ecall   ; thread complete`);
    lines.push('');
  }
  return lines;
}

// ── Assembly generation ────────────────────────────────────────────────────────

function generateAssembly(
  source: string,
  lang: Lang,
  funcs: ASTFunc[],
  vars: Map<string, VarRecord>,
  allocs: VarAllocation[],
  mapping: MappingResult
): string {
  const lines: string[] = [];
  const allocMap = new Map<string, VarAllocation>();
  for (const a of allocs) allocMap.set(a.variable, a);

  // ── Header comment ─────────────────────────────────────────────────────────
  lines.push(`; ============================================================`);
  lines.push(`; RISC-V RV32IM + Memristor Crossbar Assembly`);
  lines.push(`; Source language : ${lang === 'c' ? 'C' : 'Python'}`);
  lines.push(`; Memory mapping  : ${mapping.registerCount} register(s), ${mapping.crossbarVarCount} crossbar var(s)`);
  lines.push(`; Crossbar layout : cols 17-61 x rows 0-63 = 2880 cells`);
  lines.push(`; Utilization     : ${mapping.utilizationFactor}% (${mapping.usedCells}/2880 cells)`);
  lines.push(`; Physical device : G_MIN=200nS, G_MAX=5.9mS, V_th=830mV, V_read<=500mV`);
  lines.push(`; ============================================================`);
  lines.push('');
  lines.push(`.text`);

  let labelIdx = 0;
  const nextLabel = (prefix: string) => `${prefix}${labelIdx++}`;

  // Register assignment helper
  function regForVar(name: string): string {
    const a = allocMap.get(name);
    if (a && a.storage === 'register' && a.register) return a.register;
    return 't0'; // fallback
  }

  function colForVar(name: string): number {
    const a = allocMap.get(name);
    if (a && a.storage === 'crossbar' && a.cells && a.cells.length > 0) return a.cells[0].col;
    return 17;
  }

  function emitExpr(e: ASTExpr, destReg: string): void {
    switch (e.ek) {
      case 'num':
        lines.push(`  li   ${destReg}, ${e.value}               ; load immediate ${e.value}`);
        break;
      case 'ident': {
        const a = allocMap.get(e.name);
        if (a && a.storage === 'register' && a.register) {
          if (a.register !== destReg)
            lines.push(`  mv   ${destReg}, ${a.register}             ; ${e.name} (reg ${a.register}) → ${destReg}`);
        } else if (a && a.storage === 'crossbar' && a.cells && a.cells.length > 0) {
          const col = a.cells[0].col;
          lines.push(`  li   t6, ${col}                      ; crossbar col for ${e.name}`);
          lines.push(`  cvld ${destReg}, zero, t6             ; load ${e.name} from crossbar[0][${col}]`);
        } else {
          lines.push(`  li   ${destReg}, 0                    ; ${e.name} (unknown/param)`);
        }
        break;
      }
      case 'index': {
        const a = allocMap.get(e.array);
        const col = a?.cells?.[0]?.col ?? 17;
        emitExpr(e.idx, 't5');
        lines.push(`  li   t6, ${col}                      ; crossbar col for ${e.array}`);
        lines.push(`  cvld ${destReg}, t5, t6               ; ${e.array}[idx] from crossbar[t5][${col}]`);
        break;
      }
      case 'binop': {
        const lAlloc = e.lhs.ek === 'ident' ? allocMap.get(e.lhs.name) : undefined;
        const rAlloc = e.rhs.ek === 'ident' ? allocMap.get(e.rhs.name) : undefined;
        const bothCrossbar = lAlloc?.storage === 'crossbar' && rAlloc?.storage === 'crossbar';

        emitExpr(e.lhs, 't3');
        emitExpr(e.rhs, 't4');

        if (bothCrossbar && (e.op === '+' || e.op === '-' || e.op === '*')) {
          const xop = e.op === '+' ? 'xadd' : e.op === '-' ? 'xsub' : 'xmul';
          lines.push(`  ${xop} ${destReg}, t3, t4              ; crossbar ${e.op} (custom-1)`);
        } else {
          const ops: Record<string, string> = {'+':'add','-':'sub','*':'mul','/':'div','==':'seq','!=':'sne','<':'slt','<=':'sle','>':'sgt','>=':'sge','&&':'and','||':'or','**':'mul'};
          const rvop = ops[e.op] ?? 'add';
          if (rvop === 'seq' || rvop === 'sne' || rvop === 'slt') {
            lines.push(`  sub  ${destReg}, t3, t4               ; compare: ${exprToStr(e.lhs)} ${e.op} ${exprToStr(e.rhs)}`);
          } else {
            lines.push(`  ${rvop.padEnd(4)} ${destReg}, t3, t4              ; ${exprToStr(e.lhs)} ${e.op} ${exprToStr(e.rhs)}`);
          }
        }
        break;
      }
      case 'unop':
        emitExpr(e.expr, destReg);
        if (e.op === '-') lines.push(`  sub  ${destReg}, zero, ${destReg}         ; negate`);
        break;
      case 'call':
        if (e.func === 'max') {
          emitExpr(e.args[0] ?? { ek: 'num', value: 0 }, 't3');
          emitExpr(e.args[1] ?? { ek: 'num', value: 0 }, 't4');
          lines.push(`  ; max(${e.args.map(exprToStr).join(', ')})`);
          lines.push(`  bge  t3, t4, .+8                 ; if t3>=t4 skip`);
          lines.push(`  mv   t3, t4`);
          lines.push(`  mv   ${destReg}, t3`);
        } else if (e.func === 'min') {
          emitExpr(e.args[0] ?? { ek: 'num', value: 0 }, 't3');
          emitExpr(e.args[1] ?? { ek: 'num', value: 0 }, 't4');
          lines.push(`  ; min(${e.args.map(exprToStr).join(', ')})`);
          lines.push(`  blt  t3, t4, .+8`);
          lines.push(`  mv   t3, t4`);
          lines.push(`  mv   ${destReg}, t3`);
        } else {
          e.args.forEach((a, i) => emitExpr(a, `a${i}`));
          lines.push(`  jal  ra, ${e.func}                  ; call ${e.func}`);
          lines.push(`  mv   ${destReg}, a0                  ; return value`);
        }
        break;
      default:
        lines.push(`  li   ${destReg}, 0                    ; (unsupported expr)`);
    }
  }

  function emitStmt(stmt: ASTStmt, depth: number = 0): void {
    switch (stmt.kind) {
      case 'decl': {
        const a = allocMap.get(stmt.name);
        if (stmt.size && stmt.size > 0) {
          const col = a?.cells?.[0]?.col ?? 17;
          lines.push(`  ; --- array ${stmt.name}[${stmt.size}] → crossbar col ${col} ---`);
          for (let r = 0; r < Math.min(stmt.size, 8); r++) {
            lines.push(`  li   t0, 0                        ; ${stmt.name}[${r}] initial value`);
            lines.push(`  li   t1, ${r}                        ; row index ${r}`);
            lines.push(`  li   t2, ${col}                      ; col ${col}`);
            lines.push(`  cvst t0, t1, t2                   ; ${stmt.name}[${r}] → crossbar[${r}][${col}]`);
          }
          if (stmt.size > 8) lines.push(`  ; ... (${stmt.size - 8} more elements)`);
        } else {
          const reg = a?.register ?? 't0';
          if (stmt.init) {
            emitExpr(stmt.init, reg);
            lines.push(`  ; ${stmt.name} = ${exprToStr(stmt.init)} (reg ${reg})`);
          } else {
            lines.push(`  li   ${reg}, 0                    ; ${stmt.name} = 0 (reg ${reg})`);
          }
        }
        break;
      }
      case 'assign': {
        const a = allocMap.get(stmt.target);
        if (stmt.index) {
          // Array write → cvst
          const col = a?.cells?.[0]?.col ?? 17;
          emitExpr(stmt.value, 't0');
          emitExpr(stmt.index, 't1');
          lines.push(`  li   t2, ${col}                      ; col for ${stmt.target}`);
          lines.push(`  cvst t0, t1, t2                   ; ${stmt.target}[${exprToStr(stmt.index)}] = ${exprToStr(stmt.value)}`);
        } else {
          const reg = a?.register ?? 't0';
          emitExpr(stmt.value, reg);
          lines.push(`  ; ${stmt.target} = ${exprToStr(stmt.value)}`);
        }
        break;
      }
      case 'for': {
        const exitLbl = nextLabel('for_exit_');
        const loopLbl = nextLabel('for_loop_');
        emitStmt(stmt.init, depth);
        lines.push(`${loopLbl}:                              ; for-loop start`);
        // Evaluate condition; branch to exit if false
        const condStr = exprToStr(stmt.cond);
        if (stmt.cond.ek === 'binop') {
          emitExpr(stmt.cond.lhs, 't3');
          emitExpr(stmt.cond.rhs, 't4');
          const bmap: Record<string,string> = {'<':'bge','<=':'blt','>':`bge`,'>=':'blt','==':'bne','!=':'beq'};
          const br = bmap[stmt.cond.op] ?? 'bne';
          // invert: exit when condition is false
          const invMap: Record<string,string> = {'<':'bge','<=':'blt','>':'ble','>=':'blt','==':'bne','!=':'beq'};
          lines.push(`  ${(invMap[stmt.cond.op]??'bge').padEnd(4)} t3, t4, ${exitLbl}       ; exit if !(${condStr})`);
        } else {
          emitExpr(stmt.cond, 't3');
          lines.push(`  beq  t3, zero, ${exitLbl}           ; exit if !(${condStr})`);
        }
        stmt.body.forEach(s => emitStmt(s, depth + 1));
        emitStmt(stmt.step, depth);
        lines.push(`  jal  zero, ${loopLbl}               ; loop back`);
        lines.push(`${exitLbl}:                             ; for-loop end`);
        break;
      }
      case 'foreach': {
        const exitLbl = nextLabel('fe_exit_');
        const loopLbl = nextLabel('fe_loop_');
        const iterReg = regForVar(stmt.varName) !== 't0' ? regForVar(stmt.varName) : 's0';
        lines.push(`  li   ${iterReg}, 0                   ; ${stmt.varName} = 0`);
        lines.push(`${loopLbl}:                              ; foreach start`);
        emitExpr(stmt.iterable, 't4');
        lines.push(`  bge  ${iterReg}, t4, ${exitLbl}         ; exit if ${stmt.varName} >= bound`);
        stmt.body.forEach(s => emitStmt(s, depth + 1));
        lines.push(`  addi ${iterReg}, ${iterReg}, 1            ; ${stmt.varName}++`);
        lines.push(`  jal  zero, ${loopLbl}               ; loop back`);
        lines.push(`${exitLbl}:                             ; foreach end`);
        break;
      }
      case 'if': {
        const elseLbl = nextLabel('if_else_');
        const endLbl  = nextLabel('if_end_');
        if (stmt.cond.ek === 'binop') {
          emitExpr(stmt.cond.lhs, 't3');
          emitExpr(stmt.cond.rhs, 't4');
          const invMap: Record<string,string> = {'<':'bge','<=':'blt','>':'ble','>=':'blt','==':'bne','!=':'beq'};
          lines.push(`  ${(invMap[stmt.cond.op]??'bge').padEnd(4)} t3, t4, ${elseLbl}       ; if !(${exprToStr(stmt.cond)}) goto else`);
        } else {
          emitExpr(stmt.cond, 't3');
          lines.push(`  beq  t3, zero, ${elseLbl}          ; if !(${exprToStr(stmt.cond)}) goto else`);
        }
        stmt.then.forEach(s => emitStmt(s, depth + 1));
        if (stmt.els.length > 0) {
          lines.push(`  jal  zero, ${endLbl}               ; skip else`);
          lines.push(`${elseLbl}:                            ; else block`);
          stmt.els.forEach(s => emitStmt(s, depth + 1));
          lines.push(`${endLbl}:                             ; end if`);
        } else {
          lines.push(`${elseLbl}:                            ; end if (no else)`);
        }
        break;
      }
      case 'return': {
        if (stmt.value) emitExpr(stmt.value, 'a0');
        lines.push(`  ecall                              ; thread complete / return`);
        break;
      }
      case 'expr':
        emitExpr(stmt.expr, 't0');
        break;
      case 'unsupported':
        lines.push(`  ; (unsupported: ${stmt.text})`);
        break;
    }
  }

  for (const fn of funcs) {
    lines.push('');
    lines.push(`${fn.name}:                              ; function ${fn.name}`);
    // Load parameters
    fn.params.forEach((p, i) => {
      lines.push(`  ; param ${p.name} in a${i}${p.isPtr ? ' (pointer/array)' : ''}`);
    });
    lines.push('');
    fn.body.forEach(s => emitStmt(s));
    if (!fn.body.some(s => s.kind === 'return')) {
      lines.push(`  ecall                              ; thread complete`);
    }
  }

  // ── Memory map section ─────────────────────────────────────────────────────
  lines.push('');
  lines.push('; ============================================================');
  lines.push('; .memory_map — variable → crossbar cell assignments');
  lines.push('; ============================================================');
  for (const a of allocs) {
    if (a.storage === 'register') {
      lines.push(`;   ${a.variable.padEnd(12)} → register ${a.register}  (scalar, ${a.size} word)`);
    } else if (a.storage === 'crossbar' && a.cells) {
      const first = a.cells[0];
      const last  = a.cells[a.cells.length - 1];
      if (a.cells.length === 1) {
        lines.push(`;   ${a.variable.padEnd(12)} → crossbar[${first.row}][${first.col}]   (size=${a.size})`);
      } else {
        lines.push(`;   ${a.variable.padEnd(12)} → crossbar[${first.row}..${last.row}][${first.col}..${last.col}]   (size=${a.size})`);
      }
    }
  }
  lines.push(';');
  lines.push(`; Crossbar utilization: ${mapping.utilizationFactor}% (${mapping.usedCells}/2880 cells)`);
  lines.push(`; Registers used:       ${mapping.registerCount}/${20} GPRs`);
  lines.push('; ============================================================');

  return lines.join('\n');
}

// ── Step builders ──────────────────────────────────────────────────────────────

function buildStep1(tokens: Tok[]): CompileStep {
  const show = tokens.slice(0, 40);
  let content = show.map(t => `${t.line}:${t.kind.toUpperCase().padEnd(8)} '${t.value}'`).join('\n');
  const remaining = tokens.length - 40;
  if (remaining > 0) content += `\n... (${remaining} more)`;
  return { title: 'Step 1 — Tokenization', content };
}

function buildStep2(funcs: ASTFunc[]): CompileStep {
  const lines: string[] = [];
  function stmtLine(s: ASTStmt, depth: number): void {
    const pad = '  '.repeat(depth);
    switch (s.kind) {
      case 'decl':     lines.push(`${pad}VarDecl: ${s.vtype} ${s.name}${s.size ? `[${s.size}]` : ''}${s.init ? ' = ...' : ''}`); break;
      case 'assign':   lines.push(`${pad}Assign: ${s.target}${s.index ? '[...]' : ''} = ${exprToStr(s.value)}`); break;
      case 'for':      lines.push(`${pad}For: (${exprToStr(s.cond)})`); s.body.forEach(b => stmtLine(b, depth+1)); break;
      case 'foreach':  lines.push(`${pad}ForEach: ${s.varName} in ${exprToStr(s.iterable)}`); s.body.forEach(b => stmtLine(b, depth+1)); break;
      case 'if':       lines.push(`${pad}If: ${exprToStr(s.cond)}`); s.then.forEach(b => stmtLine(b, depth+1)); if (s.els.length) { lines.push(`${pad}Else:`); s.els.forEach(b => stmtLine(b, depth+1)); } break;
      case 'return':   lines.push(`${pad}Return: ${s.value ? exprToStr(s.value) : 'void'}`); break;
      case 'expr':     lines.push(`${pad}Expr: ${exprToStr(s.expr)}`); break;
      case 'unsupported': lines.push(`${pad}; (unsupported)`); break;
    }
  }
  for (const fn of funcs) {
    lines.push(`Function: ${fn.name}(${fn.params.map(p => `${p.type} ${p.name}${p.isPtr?'*':''}`).join(', ')}) -> ${fn.returnType}`);
    fn.params.forEach(p => lines.push(`  Param: ${p.name} (${p.type}${p.isPtr ? ', ptr' : ''})`));
    fn.body.slice(0, 50).forEach(s => stmtLine(s, 1));
    if (fn.body.length > 50) lines.push(`  ... (${fn.body.length - 50} more statements)`);
  }
  return { title: 'Step 2 — Abstract Syntax Tree', content: lines.slice(0, 50).join('\n') };
}

function buildStep3(irLines: string[]): CompileStep {
  return { title: 'Step 3 — Three-Address IR', content: irLines.slice(0, 50).join('\n') + (irLines.length > 50 ? `\n... (${irLines.length - 50} more)` : '') };
}

function buildStep4(vars: Map<string, VarRecord>): CompileStep {
  const rows: string[] = ['  Name          Kind    Def   Last  Uses'];
  for (const [, v] of vars) {
    const kind = v.isArray ? `array${v.size}` : 'scalar';
    rows.push(`  ${v.name.padEnd(14)}${kind.padEnd(8)}def:${String(v.defLine).padEnd(4)} last:${String(v.lastUseLine).padEnd(5)}uses:${v.accessCount}`);
  }
  return { title: 'Step 4 — Variable Analysis', content: rows.join('\n') };
}

function buildStep5(vars: Map<string, VarRecord>, mapping: MappingResult): CompileStep {
  const lines: string[] = [];
  lines.push(`Crossbar Data Region: cols 17-61 x rows 0-63 = 2880 cells`);
  lines.push(`Utilization: ${mapping.utilizationFactor}% (${mapping.usedCells}/2880 cells)`);
  lines.push('');
  lines.push('Register file (hot scalars):');
  for (const a of mapping.allocations) {
    if (a.storage === 'register') {
      const vr = vars.get(a.variable);
      lines.push(`  ${a.variable.padEnd(12)} → ${a.register}   (${vr?.accessCount ?? 0} uses)`);
    }
  }
  lines.push('');
  lines.push('Crossbar variable region (arrays + spilled scalars):');
  for (const a of mapping.allocations) {
    if (a.storage === 'crossbar' && a.cells && a.cells.length > 0) {
      const first = a.cells[0];
      const last  = a.cells[a.cells.length - 1];
      const vr = vars.get(a.variable);
      const isSpilled = vr && !vr.isArray;
      if (a.cells.length === 1) {
        lines.push(`  ${a.variable.padEnd(12)} → crossbar[${first.row}][${first.col}]   (size=${a.size}${isSpilled ? ', spilled' : ''})`);
      } else {
        lines.push(`  ${a.variable.padEnd(12)} → crossbar[${first.row}..${last.row}][${first.col}..${last.col}]   (size=${a.size})`);
      }
    }
  }
  // Count lifetime reuse pairs
  const varList = Array.from(vars.values());
  let reusePairs = 0;
  for (let i = 0; i < varList.length; i++) {
    for (let j = i + 1; j < varList.length; j++) {
      if (varList[i].lastUseLine < varList[j].defLine || varList[j].lastUseLine < varList[i].defLine) {
        reusePairs++;
      }
    }
  }
  lines.push('');
  lines.push(`Lifetime reuse: ${reusePairs} variable pair(s) share cells (non-overlapping lifetimes)`);
  return { title: 'Step 5 — Memory Mapping', content: lines.join('\n') };
}

function buildStep6(assembly: string): CompileStep {
  return { title: 'Step 6 — Assembly Generation', content: assembly };
}

// ── Default example sources ────────────────────────────────────────────────────

export const DEFAULT_C_SOURCE = `// Vector add kernel (C)
// Adds two arrays element-wise and stores result
int vec_add(int* a, int* b, int* out, int n) {
    int tid = 0;
    int i = 0;
    int val_a = 0;
    int val_b = 0;
    int sum = 0;
    for (i = 0; i < n; i++) {
        val_a = a[i];
        val_b = b[i];
        sum = val_a + val_b;
        out[i] = sum;
    }
    return 0;
}`;

export const DEFAULT_PY_SOURCE = `# ReLU activation (Python)
# Applies ReLU: max(0, x) element-wise

def relu(x, out, n):
    i = 0
    val = 0
    zero = 0
    for i in range(n):
        val = x[i]
        if val < zero:
            out[i] = 0
        else:
            out[i] = val`;

// ── Main export ────────────────────────────────────────────────────────────────

export function crossCompile(source: string, lang: Lang): CrossCompileResult {
  const errors: string[] = [];
  const emptyMapping: MappingResult = {
    allocations: [], utilizationFactor: 0, usedCells: 0,
    totalCells: 2880, registerCount: 0, crossbarVarCount: 0,
    heatmap: Array.from({ length: 64 }, () => new Array(45).fill(0)),
  };

  try {
    // ── Step 1: Tokenize ──────────────────────────────────────────────────────
    const tokens = tokenize(source, lang);
    const step1  = buildStep1(tokens);

    // ── Parse ─────────────────────────────────────────────────────────────────
    let funcs: ASTFunc[] = [];
    try {
      funcs = parse([...tokens], lang);
    } catch (e) {
      errors.push(`Parse error: ${(e as Error).message}`);
      funcs = [{ name: 'main', params: [], body: [], returnType: 'void' }];
    }

    // ── Step 2: AST ───────────────────────────────────────────────────────────
    const step2 = buildStep2(funcs);

    // ── Variable analysis ─────────────────────────────────────────────────────
    const vars = new Map<string, VarRecord>();
    for (const fn of funcs) {
      fn.params.forEach((p, i) => {
        vars.set(p.name, { name: p.name, isArray: p.isPtr, size: p.isPtr ? 64 : 1, defLine: 0, lastUseLine: 1000, accessCount: 2 });
      });
      analyzeStmts(fn.body, vars, 0);
    }

    // ── Step 4: Variable analysis ─────────────────────────────────────────────
    const step4 = buildStep4(vars);

    // ── Memory mapping ────────────────────────────────────────────────────────
    const varInfos: VariableInfo[] = Array.from(vars.values()).map(v => ({
      name: v.name, isArray: v.isArray, size: v.size,
      defLine: v.defLine, lastUseLine: v.lastUseLine, accessCount: v.accessCount,
    }));
    const mapping = mapVariables(varInfos);

    // ── Step 5: Memory mapping ────────────────────────────────────────────────
    const step5 = buildStep5(vars, mapping);

    // ── IR generation ─────────────────────────────────────────────────────────
    const irLines = generateIR(funcs, vars, mapping.allocations);
    const step3   = buildStep3(irLines);

    // ── Assembly generation ───────────────────────────────────────────────────
    const assembly = generateAssembly(source, lang, funcs, vars, mapping.allocations, mapping);
    const step6    = buildStep6(assembly);

    return {
      steps: [step1, step2, step3, step4, step5, step6],
      assembly,
      mapping,
      errors,
    };
  } catch (e) {
    errors.push(`Compiler error: ${(e as Error).message}`);
    return {
      steps: [
        { title: 'Step 1 — Tokenization',      content: '(error)' },
        { title: 'Step 2 — AST',               content: '(error)' },
        { title: 'Step 3 — Three-Address IR',  content: '(error)' },
        { title: 'Step 4 — Variable Analysis', content: '(error)' },
        { title: 'Step 5 — Memory Mapping',    content: '(error)' },
        { title: 'Step 6 — Assembly',          content: '(error)' },
      ],
      assembly: '; compilation failed\n',
      mapping: emptyMapping,
      errors,
    };
  }
}

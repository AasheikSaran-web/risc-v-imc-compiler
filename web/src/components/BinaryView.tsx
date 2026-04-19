import { Instruction } from '../compiler/types';

interface BinaryViewProps {
  instructions: Instruction[];
  highlightAddr?: number;
}

/** Color-coded 32-bit RISC-V instruction view */
export function BinaryView({ instructions, highlightAddr }: BinaryViewProps) {
  if (instructions.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        No instructions generated yet. Write a kernel above to compile.
      </div>
    );
  }

  return (
    <div style={{ overflow: 'auto', fontSize: '12px', fontFamily: 'monospace' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#111', color: '#888', textAlign: 'left' }}>
            <th style={{ padding: '6px 8px', width: '40px' }}>Addr</th>
            <th style={{ padding: '6px 8px', width: '90px' }}>Hex (32-bit)</th>
            <th style={{ padding: '6px 8px', width: '240px' }}>RISC-V Binary</th>
            <th style={{ padding: '6px 8px' }}>Assembly</th>
          </tr>
        </thead>
        <tbody>
          {instructions.map((inst) => {
            const isHighlighted = inst.addr === highlightAddr;
            const raw = (parseInt(inst.hex.replace('0x', ''), 16) >>> 0);
            const opcode  = raw & 0x7F;
            const rd      = (raw >> 7) & 0x1F;
            const funct3  = (raw >> 12) & 0x7;
            const rs1     = (raw >> 15) & 0x1F;
            const upper   = (raw >>> 20) & 0xFFF;

            return (
              <tr
                key={inst.addr}
                style={{
                  background: isHighlighted ? '#1e2e3e' : 'transparent',
                  borderLeft: isHighlighted ? '3px solid #4ec9b0' : '3px solid transparent',
                  transition: 'background 0.15s',
                }}
              >
                <td style={{ padding: '4px 8px', color: '#666' }}>{inst.addr}</td>
                <td style={{ padding: '4px 8px', color: '#b5cea8' }}>{inst.hex}</td>
                <td style={{ padding: '4px 8px' }}>
                  <RVBits opcode={opcode} rd={rd} funct3={funct3} rs1={rs1} upper={upper} />
                </td>
                <td style={{ padding: '4px 8px', color: '#e0e0e0' }}>{inst.asm}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Render 32 RISC-V bits with color coding per field */
function RVBits({
  opcode, rd, funct3, rs1, upper,
}: {
  opcode: number; rd: number; funct3: number; rs1: number; upper: number;
}) {
  const b7  = (n: number, w: number) => n.toString(2).padStart(w, '0');

  return (
    <span style={{ letterSpacing: '0.5px' }}>
      {/* [31:20] imm/rs2/funct7 */}
      <span style={{ color: '#d19a66' }} title="imm[11:0] / rs2+funct7 [31:20]">
        {b7(upper, 12)}
      </span>
      {' '}
      {/* [19:15] rs1 */}
      <span style={{ color: '#98c379' }} title="rs1 [19:15]">
        {b7(rs1, 5)}
      </span>
      {' '}
      {/* [14:12] funct3 */}
      <span style={{ color: '#e5c07b' }} title="funct3 [14:12]">
        {b7(funct3, 3)}
      </span>
      {' '}
      {/* [11:7] rd */}
      <span style={{ color: '#61afef' }} title="rd [11:7]">
        {b7(rd, 5)}
      </span>
      {' '}
      {/* [6:0] opcode */}
      <span style={{ color: '#e06c75' }} title="opcode [6:0]">
        {b7(opcode, 7)}
      </span>
    </span>
  );
}

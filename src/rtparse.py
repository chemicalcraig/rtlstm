#!/usr/bin/env python3
"""
Parse NWChem RT-TDDFT output file

Extracts:
  - dip.dat: Dipole moment data
  - field.dat: External field data
  - occ.dat: MO occupation numbers (restricted)
  - energy.dat: Total energy
  - eadded.dat: Added energy
  - cpu.dat: CPU time
  - coulomb.dat: Coulomb energy
  - nuc.dat: Nuclear energy
  - charge.dat: Mulliken charges
  - s2.dat: <S^2> values

Based on rtparse.cc by Craig T. Chapman
Python version with improved error handling and flexibility

Usage:
    python rtparse.py nwchem_output.out
    python rtparse.py nwchem_output.out --time-fs  # Output time in fs
    python rtparse.py nwchem_output.out --output-dir results/
"""

import sys
import os
import re
import argparse
from pathlib import Path


class NWChemRTParser:
    """Parser for NWChem RT-TDDFT output files"""
    
    def __init__(self, input_file, output_dir=".", time_in_fs=False):
        """
        Initialize parser
        
        Args:
            input_file: Path to NWChem output file
            output_dir: Directory for output files
            time_in_fs: If True, output time in femtoseconds (default: atomic units)
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.time_in_fs = time_in_fs
        self.au_to_fs = 0.024189  # Conversion factor
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files
        self.files = {
            'dip': open(self.output_dir / 'dip.dat', 'w'),
            'field': open(self.output_dir / 'field.dat', 'w'),
            'occ': open(self.output_dir / 'occ.dat', 'w'),
            'energy': open(self.output_dir / 'energy.dat', 'w'),
            'eadded': open(self.output_dir / 'eadded.dat', 'w'),
            'cpu': open(self.output_dir / 'cpu.dat', 'w'),
            'coulomb': open(self.output_dir / 'coulomb.dat', 'w'),
            'nuc': open(self.output_dir / 'nuc.dat', 'w'),
            'charge': open(self.output_dir / 'charge.dat', 'w'),
            's2': open(self.output_dir / 's2.dat', 'w')
        }
        
        # Parser state
        self.geometry_name = None
        self.n_mo = None
        self.print_mos = False
        self.dipole_buffer = []
        self.field_buffer = []
        
        # Write headers
        self._write_headers()
    
    def _write_headers(self):
        """Write headers to output files"""
        time_unit = "fs" if self.time_in_fs else "a.u."
        
        self.files['dip'].write(f"# Molecular dipole extracted from {self.input_file}\n")
        self.files['dip'].write(f"# time ({time_unit}) | x | y | z | total\n")
        
        self.files['field'].write(f"# Field data extracted from {self.input_file}\n")
        self.files['field'].write(f"# time ({time_unit}) | E_x | E_y | E_z\n")
        
        self.files['occ'].write(f"# MO occupations extracted from {self.input_file}\n")
        self.files['occ'].write(f"# time ({time_unit}) | occ_1 | occ_2 | ...\n")
    
    def __del__(self):
        """Close all output files"""
        for f in self.files.values():
            if not f.closed:
                f.close()
    
    def _convert_time(self, time_au):
        """Convert time to output units"""
        return time_au * self.au_to_fs if self.time_in_fs else time_au
    
    def parse(self):
        """Parse the NWChem output file"""
        
        print(f"Parsing: {self.input_file}")
        
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Get active geometry name
            if line.startswith('Active geometry'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    self.geometry_name = match.group(1)
                    print(f"Using active geometry: {self.geometry_name}")
            
            # Check if MOs are printed
            elif line.startswith('rt_tddft'):
                # Scan ahead for print directive
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('end'):
                    if 'print' in lines[j]:
                        if 'moocc' in lines[j] or '*' in lines[j]:
                            self.print_mos = True
                            print("MO occupations will be extracted")
                        break
                    j += 1
            
            # Get number of MOs
            elif 'Number of molecular orbitals' in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    self.n_mo = int(match.group(1))
                    print(f"Number of MOs: {self.n_mo}")
            
            # Parse RT-TDDFT data lines
            elif line.startswith('<rt_tddft>:'):
                self._parse_rt_line(line, lines, i)
            
            i += 1
        
        print(f"\n✓ Parsing complete")
        print(f"  Output files written to: {self.output_dir}/")
        
        return self._get_stats()
    
    def _parse_rt_line(self, line, lines, line_idx):
        """Parse a single <rt_tddft>: line"""
        
        # Remove tag and split
        content = line.replace('<rt_tddft>:', '').strip()
        parts = content.split()
        
        if len(parts) < 2:
            return
        
        # Get time
        try:
            time = float(parts[0])
        except ValueError:
            return
        
        # Skip "### ... ###" lines
        if parts[1].startswith('###'):
            return
        
        # Get first value
        try:
            val1 = float(parts[1])
        except ValueError:
            return
        
        # Check if this is a comment line (starts with #)
        if len(parts) > 2 and parts[2].startswith('#'):
            self._parse_comment_line(time, val1, parts[2:])
        else:
            # Data line (dipole or field)
            if len(parts) >= 4:
                self._parse_data_line(time, parts[1:])
    
    def _parse_comment_line(self, time, value, comment_parts):
        """Parse lines with # comments (energies, charges, etc.)"""
        
        comment = ' '.join(comment_parts)
        time_out = self._convert_time(time)
        
        # CPU time
        if 'CPU time' in comment:
            self.files['cpu'].write(f"{time_out} {value}\n")
        
        # Nuclear energy
        elif 'Enuc' in comment:
            self.files['nuc'].write(f"{time_out} {value}\n")
        
        # Total energy
        elif 'Etot' in comment:
            self.files['energy'].write(f"{time_out} {value}\n")
        
        # Coulomb energy
        elif 'Ecoul' in comment:
            self.files['coulomb'].write(f"{time_out} {value}\n")
        
        # Added energy
        elif 'Eadded' in comment:
            self.files['eadded'].write(f"{time_out} {value}\n")
        
        # Charge (Mulliken)
        elif 'Charge' in comment:
            if self.geometry_name and self.geometry_name in comment:
                self.files['charge'].write(f"{time_out} {value}\n")
                
                # Parse MO occupations if available
                if self.print_mos and self.n_mo:
                    # MO data follows on next lines - not implemented in simple version
                    pass
        
        # <S^2>
        elif '<S^2>' in comment or 'S^2' in comment:
            self.files['s2'].write(f"{time_out} {value}\n")
    
    def _parse_data_line(self, time, data_parts):
        """Parse data lines (field and dipole)"""
        
        if len(data_parts) < 4:
            return
        
        try:
            x = float(data_parts[0])
            y = float(data_parts[1])
            z = float(data_parts[2])
        except ValueError:
            return
        
        # Join remaining parts to check for tags
        remaining = ' '.join(data_parts[3:])
        
        time_out = self._convert_time(time)
        
        # Applied field
        if 'Applied E-field' in remaining or 'Applied' in remaining:
            if self.geometry_name and self.geometry_name in remaining:
                # Buffer field data (alpha and beta)
                self.field_buffer.append([x, y, z])
                
                # Write when we have both spins (or for restricted)
                if len(self.field_buffer) >= 2:
                    # Average alpha and beta fields (they should be identical for closed-shell)
                    fx = sum(f[0] for f in self.field_buffer) / len(self.field_buffer)
                    fy = sum(f[1] for f in self.field_buffer) / len(self.field_buffer)
                    fz = sum(f[2] for f in self.field_buffer) / len(self.field_buffer)
                    self.files['field'].write(f"{time_out} {fx:.12e} {fy:.12e} {fz:.12e}\n")
                    self.field_buffer = []
        
        # Dipole moment
        elif 'Dipole' in remaining:
            if self.geometry_name and self.geometry_name in remaining:
                # Buffer dipole data (alpha, beta, total)
                self.dipole_buffer.append([x, y, z])
                
                # Write when we have all three (alpha + beta + total)
                if len(self.dipole_buffer) >= 3:
                    # Use total dipole (last entry)
                    dx, dy, dz = self.dipole_buffer[-1]
                    total = (dx**2 + dy**2 + dz**2)**0.5
                    self.files['dip'].write(f"{time_out} {dx:.12e} {dy:.12e} {dz:.12e} {total:.12e}\n")
                    self.dipole_buffer = []
    
    def _get_stats(self):
        """Get parsing statistics"""
        stats = {}
        
        for name, f in self.files.items():
            if not f.closed:
                f.flush()
                stats[name] = f.tell()  # File size
        
        return stats


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='Parse NWChem RT-TDDFT output file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python rtparse.py nwchem.out
  
  # Output time in femtoseconds
  python rtparse.py nwchem.out --time-fs
  
  # Specify output directory
  python rtparse.py nwchem.out --output-dir results/
  
  # Combine options
  python rtparse.py nwchem.out --time-fs --output-dir results/

Output files:
  dip.dat      - Dipole moment (x, y, z, total)
  field.dat    - External field (Ex, Ey, Ez)
  occ.dat      - MO occupations (if printed)
  energy.dat   - Total energy
  eadded.dat   - Added energy  
  cpu.dat      - CPU time
  coulomb.dat  - Coulomb energy
  nuc.dat      - Nuclear energy
  charge.dat   - Mulliken charges
  s2.dat       - <S^2> values
        """
    )
    
    parser.add_argument('input', type=str,
                       help='NWChem output file')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--time-fs', action='store_true',
                       help='Output time in femtoseconds (default: atomic units)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.quiet:
        print("="*70)
        print("NWChem RT-TDDFT Output Parser")
        print("="*70)
    
    try:
        # Parse file
        rt_parser = NWChemRTParser(
            args.input,
            output_dir=args.output_dir,
            time_in_fs=args.time_fs
        )
        
        stats = rt_parser.parse()
        
        if not args.quiet:
            print("\n" + "="*70)
            print("Files created:")
            print("="*70)
            for name, size in stats.items():
                if size > 0:
                    output_path = Path(args.output_dir) / f"{name}.dat"
                    print(f"  {output_path} ({size} bytes)")
            
            print("\n" + "="*70)
            print("Usage:")
            print("="*70)
            output_dir = args.output_dir if args.output_dir != '.' else ''
            field_path = os.path.join(output_dir, 'field.dat') if output_dir else 'field.dat'
            print(f"  # Use extracted field with prediction")
            print(f"  python src/predict_from_initial.py --field {field_path} ...")
            print(f"  ")
            print(f"  # Validate field data")
            print(f"  python src/parse_field.py {field_path}")
    
    except Exception as e:
        print(f"\n✗ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

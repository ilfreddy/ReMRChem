import glob
import os
import re

import numpy as np



input_files = glob.glob('../Input_Making/input__*')

def parse_filename(filename):
    # Remove path and extension
    base = os.path.basename(filename)
    name = base.replace('.txt', '')
    # Extract part after '__'
    parts = name.split('__', 1)
    if len(parts) != 2:
        return None
    suffix = parts[1]
    # Match: 2 chars, 3 digits, up to 4 chars
    match = re.match(r'^([A-Za-z]{2})(\d)(\d)(\d)([A-Za-z0-9]{0,4})$', suffix)
    if match:
        two_chars = match.group(1)
        digits = [match.group(2), match.group(3), match.group(4)]
        last_str = match.group(5)
        return {
            'two_chars': two_chars,
            'digits': digits,
            'last_str': last_str
        }
    else:
        return None

for file in input_files:
    result = parse_filename(file)
    if result:
        #print(f"File: {file}")
        #print(f"  Atom: {result['two_chars']}")
        #print(f"  Digit 1: {result['digits'][0]}")
        #print(f"  Digit 2: {result['digits'][1]}")
        #print(f"  Digit 3: {result['digits'][2]}")
        #print(f"  Last string: {result['last_str']}")
    
        Atom_label = result['two_chars']
        centered = int(result['digits'][0])
        D2 = int(result['digits'][1])
        mLog_prec = int(result['digits'][2])
        Derivative_type = result['last_str']
    
        # Assign Z by reading ../Z.txt and matching Atom_label
        # Z.txt is like a dictionary: each line "Atom_label Z"
        Z_dict = {}
        with open('../Z.txt', 'r') as zfile:
            for line in zfile:
                columns = line.strip().split()
                if len(columns) >= 2:
                    Z_dict[columns[0]] = columns[1]
        Z = int(Z_dict.get(Atom_label))
        #print(f"  Z: {Z}")

        thr = mLog_prec - 1
        order = mLog_prec + 3

        box = int(np.ceil(50 /Z))
        # Find corresponding output file
        output_file = f"../outputs/output__{Atom_label}{centered}{D2}{mLog_prec}{Derivative_type}.txt"
        energy = None
        print(f"  Output file: {output_file}")
        if os.path.exists(output_file):
            with open(output_file, 'r') as ofile:
            
                for line in ofile:
                    if line.strip().startswith("Dirac Energy ="):
                        parts = line.strip().split("=")
                        if len(parts) == 2:
                            energy = parts[1].strip()
                            break
        if energy is None:
            energy = "N/A"
        print(Atom_label, Z, thr, order, box, centered, D2, mLog_prec, Derivative_type, energy)
        #with open('Data.txt', 'a') as datafile:
        #    datafile.write(f"{Atom_label}\t{Z}\t{thr}\t{order}\t{box}\t{centered}\t{D2}\t{mLog_prec}\t{Derivative_type}\t{energy}\n")



    else:
        print(f"File: {file} does not match the expected pattern.")





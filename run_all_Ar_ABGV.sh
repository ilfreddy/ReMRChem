#!/bin/bash

INPUT_DIR="Input_Making"
SCRIPT="test.py"

# === Definisci l’ordine dei blocchi numerici ===
ORDER=(004 014 005 015 006 016 007 017 008 018 104 114 105 115 106 116 107 117 108 118)
#ORDER=(004 014 005 015 006 016 007 017 008 018)
#ORDER=(104 114 105 115 106 116 107 117 108 118)

echo "=== Lancio dei calcoli in ordine definito ==="
for num in "${ORDER[@]}"; do
    echo "input__Ar${num}ABGV.txt"
done     


for num in "${ORDER[@]}"; do
    echo ">>> Elaboro tutti i file contenenti Ar${num}ABGV"
    for file in "$INPUT_DIR"/input__Ar${num}ABGV.txt; do
        # verifica che esista davvero (il glob potrebbe non espandersi)
        [[ -f "$file" ]] || continue
        echo "Now executing: python $SCRIPT $file"
        python "$SCRIPT" "$file"
    done
done

#curl -X POST https://api.pushcut.io/MQFwODx_F6l1zq76_NHmN/notifications/notify_calc 



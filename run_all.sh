#!/bin/bash
# Run every experiment in order. Usage: ./run_all.sh            (all milestones)
#                                        ./run_all.sh m1 m3      (only some)
# Needs: ngspice, iverilog/vvp, python3 with numpy + matplotlib, and the SKY130 PDK
#        (PDK_ROOT/PDK set, or SKY130_LIB pointing at sky130.lib.spice).
set -e
cd "$(dirname "$0")"

export PDK_ROOT=${PDK_ROOT:-$HOME/.ciel}
export PDK=${PDK:-sky130A}
LIB=${SKY130_LIB:-$PDK_ROOT/$PDK/libs.tech/ngspice/sky130.lib.spice}

fail=0
for t in ngspice iverilog vvp python3; do
  command -v $t >/dev/null || { echo "missing: $t"; fail=1; }
done
[ -f "$LIB" ] || { echo "SKY130 model library not found: $LIB"; fail=1; }
python3 -c "import numpy, matplotlib" 2>/dev/null || { echo "missing python packages: pip install -r requirements.txt"; fail=1; }
[ $fail = 0 ] || exit 1
echo "ngspice : $(command -v ngspice)"
echo "models  : $LIB"

steps=${*:-m1 m2 m3 m4 m5}
cd scripts
for s in $steps; do
  case $s in
    m1) f=m1_dram.py ;;
    m2) f=m2_dram.py ;;
    m3) f=m3_sram.py ;;
    m4) f=m4_monte_carlo.py ;;
    m5) f=m5_rnm.py ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
  echo "=== $s: $f"
  start=$(date +%s)
  python3 $f > ../results/$s.log 2>&1 || { echo "$s failed, see results/$s.log"; exit 1; }
  echo "    done in $(( $(date +%s) - start )) s"
done
python3 summary.py
echo "All done: plots/ and results/summary.md"

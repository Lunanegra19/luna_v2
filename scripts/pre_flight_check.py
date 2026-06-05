import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pre_flight.core import run_all
import pre_flight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Luna V1 Pre-Flight Check v3.3")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--section", default=None,
                        help="Seccion(es) a ejecutar. Una sola ('env') o multiples coma-separadas ('env,v5_bugs'). "
                             "Opciones: legacy,sop,temporal,architecture,artifacts,code,data,math,consistency,env,v5_bugs,r14_fixes")
    args = parser.parse_args()
    ok = run_all(fail_fast=args.fail_fast, verbose=args.verbose,
                 section_filter=args.section)
    sys.exit(0 if ok else 1)

import sys
import os
from pathlib import Path

_ROOT = Path(r"c:\Users\Usuario\Desktop\ia\luna_v2")
sys.path.insert(0, str(_ROOT))

from config.settings import cfg
from loguru import logger
import scripts.evaluate_ensemble_wfb as eval_script

def run_test(name, seeds):
    print(f"\n==========================================")
    print(f"RUNNING TEST: {name}")
    print(f"SEEDS: {seeds}")
    print(f"==========================================\n")
    
    # Monkeypatch
    original_seeds = cfg.wfb.active_seeds
    cfg.wfb.active_seeds = seeds
    
    try:
        eval_script.main()
    except Exception as e:
        print(f"Error during eval: {e}")
    finally:
        cfg.wfb.active_seeds = original_seeds

if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    approved_seeds = [100, 2025, 23863, 54260, 97197, 82697, 68222]
    all_seeds_this_run = [42, 100, 777, 1337, 2025, 23863, 54260, 97197, 82697, 79870, 68222]
    
    run_test('ONLY APPROVED SEEDS', approved_seeds)
    run_test('ALL COMPLETED SEEDS', all_seeds_this_run)

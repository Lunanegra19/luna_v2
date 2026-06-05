import os
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
reports_dir = _ROOT / "data" / "reports"

def audit_json():
    print("="*90)
    print("      DIAGNOSTIC AUDIT: DETAILED GATES AND VERDICTS FOR THE MAY 22 MULTI-SEED RUN      ")
    print("="*90)
    
    verdicts = list(reports_dir.glob("2026-05-22_T*_seed*_FINAL_statistical_verdict.json"))
    if not verdicts:
        print("No May 22 verdict files found in data/reports.")
        return
        
    for vf in sorted(verdicts):
        print(f"\nFile: {vf.name}")
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                v = json.load(f)
                
            # Parse correct keys from JSON structure
            deploy_approved = v.get('deploy_approved', False)
            metrics = v.get('metrics', {})
            stat_audit = v.get('statistical_audit', {})
            flags = v.get('flags', {})
            run_id = v.get('run_id', 'UNKNOWN')
            
            # Extract values
            trades = metrics.get('total_trades', 0)
            wr = metrics.get('win_rate', 0.0) * 100
            ret = metrics.get('total_return_pct', 0.0) * 100
            sharpe = metrics.get('sharpe_crudo', 0.0)
            max_dd = metrics.get('max_drawdown_pct', 0.0) * 100
            
            dsr = stat_audit.get('dsr', 0.0)
            pbo = stat_audit.get('estimated_pbo', 0.0) * 100
            binomial_p = stat_audit.get('binomial_p_value', 1.0)
            
            pass_dsr = flags.get('pass_dsr', False)
            pass_trades = flags.get('pass_trades', False)
            pass_dd = flags.get('pass_dd', False)
            pass_pbo = flags.get('pass_pbo', False)
            pass_binomial = flags.get('pass_binomial', False)
            
            print(f"Run ID:  {run_id}")
            print(f"Status:  {'APPROVED' if deploy_approved else 'REJECTED'}")
            print(f"Metrics: Trades={trades:3d} | WR={wr:5.2f}% | Return={ret:8.4f}% | Sharpe={sharpe:6.4f} | Max DD={max_dd:7.4f}%")
            print(f"Audit:   DSR={dsr:6.4f} ({'PASS' if pass_dsr else 'FAIL'}) | PBO={pbo:5.2f}% ({'PASS' if pass_pbo else 'FAIL'}) | Binomial p={binomial_p:6.4f} ({'PASS' if pass_binomial else 'FAIL'})")
            print(f"Gates:   Trades Pass: {pass_trades} | DD Pass: {pass_dd}")
            
            # If there was a pipeline status/reason, print it
            pipeline = v.get('signal_pipeline', {})
            if pipeline:
                print(f"W5 Pipeline Status: {pipeline.get('status')} | Reason: {pipeline.get('reason')}")


            
        except Exception as e:
            print(f"Error parsing file: {e}")

if __name__ == "__main__":
    audit_json()

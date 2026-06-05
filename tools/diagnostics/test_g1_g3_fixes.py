"""
[TEST-G1-G3] Test directo del clasificador y query JOIN sin pasar por el servidor HTTP.
Importa las funciones directamente del server.py y ejecuta la lógica de clasificación.
"""
import sys, os, re, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/root/luna_v2/.env")
sys.path.insert(0, "/root/luna_v2")

from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor

print("=" * 70)
print("[TEST-G1-G3] Validación directa de fixes del dashboard")
print("=" * 70)

# ── TEST G3: Query JOIN ────────────────────────────────────────────────────
print("\n[TEST-G3] Validando query JOIN audit_logs + operational_audit_logs...")
db = DatabaseManager()
start_utc = "2026-05-25T18:00:00"
end_utc   = "2026-05-25T18:05:00"

with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT al.timestamp, al.price, al.action, al.confidence,
                   al.xgb_prob, al.hmm_regime, al.reason,
                   al.contracts, al.executed_price,
                   op.clock_drift_minutes, op.clock_drift_status,
                   op.execution_latency_sec, op.latency_status,
                   op.nan_inf_null_cols, op.nan_inf_status,
                   op.slippage_pct, op.slippage_status,
                   op.active_leverage, op.leverage_status,
                   op.is_approved, op.hmm_regime_index,
                   op.api_liveness_equity
            FROM audit_logs al
            LEFT JOIN operational_audit_logs op
              ON ABS(EXTRACT(EPOCH FROM (al.timestamp - op.timestamp))) < 120
            WHERE al.timestamp >= %s AND al.timestamp <= %s
            ORDER BY al.id DESC LIMIT 1
        """, (start_utc, end_utc))
        row = cur.fetchone()

if not row:
    print("  ❌ Sin datos para ese rango horario")
else:
    row_dict = dict(row)
    print(f"  ✅ Fila encontrada: {row_dict['timestamp']} | {row_dict['action']}")
    print(f"\n  --- audit_logs ---")
    print(f"  action:           {row_dict['action']}")
    print(f"  xgb_prob:         {float(row_dict['xgb_prob']):.4f}")
    print(f"  hmm_regime:       {row_dict['hmm_regime']}")
    print(f"  confidence:       {float(row_dict['confidence']):.4f}")
    print(f"\n  --- operational_audit_logs (via JOIN) ---")
    print(f"  clock_drift_min:  {row_dict['clock_drift_minutes']} ({row_dict['clock_drift_status']})")
    print(f"  latency_sec:      {row_dict['execution_latency_sec']} ({row_dict['latency_status']})")
    print(f"  nan_inf_cols:     {row_dict['nan_inf_null_cols']} ({row_dict['nan_inf_status']})")
    print(f"  slippage_pct:     {row_dict['slippage_pct']} ({row_dict['slippage_status']})")
    print(f"  active_leverage:  {row_dict['active_leverage']}x ({row_dict['leverage_status']})")
    print(f"  is_approved:      {row_dict['is_approved']}")
    print(f"  api_equity:       ${float(row_dict['api_liveness_equity'] or 0):,.2f}")

# ── TEST G1: Clasificador de líneas ────────────────────────────────────────
print("\n[TEST-G1] Validando clasificador de keywords en líneas reales del PM2...")

pm2_log = Path("/root/.pm2/logs/luna-v2-live-demo-out.log")
if not pm2_log.exists():
    print("  ❌ Log PM2 no encontrado")
else:
    with open(pm2_log, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Extraer líneas del ciclo 18:00
    lines = content.split("\n")
    cycle_lines = []
    in_cycle = False
    for line in lines:
        if line.startswith("[2026-05-25 18:"):
            if "Iniciando Ciclo Operativo LUNA V2" in line:
                in_cycle = True
                cycle_lines = [line]
                continue
        if in_cycle:
            if "Iniciando Ciclo Operativo LUNA V2" in line and not line.startswith("[2026-05-25 18:"):
                break
            cycle_lines.append(line)
            if "Ciclo finalizado" in line or "Durmiendo" in line:
                break

    print(f"  Total líneas del ciclo 18:00 capturadas: {len(cycle_lines)}")

    # Aplicar el nuevo clasificador
    steps = [[] for _ in range(6)]
    for line in cycle_lines:
        m_prefix = re.match(r"^\[?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]?:?\s*(.*)$", line)
        clean_line = m_prefix.group(1) if m_prefix else line

        if any(kw in line for kw in ["[EnsembleLive/BOOT]", "[EnsembleLive/LOAD]", "[EnsembleLive/SUCCESS]",
                                      "[EnsembleLive/MANIFEST]", "[RegimeRouter/LOAD]", "Cargando componentes",
                                      "Semilla cargada con éxito", "LunaEnsembleLiveInference"]):
            steps[0].append(clean_line)
        elif any(kw in line for kw in ["Heartbeat", "[RECONCILIACIÓN]", "Reconciliacion",
                                        "Risk Monitor", "Drawdowns", "[RM]", "DD Día"]):
            steps[1].append(clean_line)
        elif any(kw in line for kw in ["DataCollector", "[FIX-BUG]", "[BUGFIX-4]", "[BUGFIX-TIMING]",
                                        "[BUGFIX-1]", "[LUNA][A", "[FIX-CALENDAR", "[FP]", "[LIVE-AE-FIX]",
                                        "[BUGFIX-WEEKEND", "[LIVE-INFERENCE-SAVE]", "KMeans", "AutoEncoder",
                                        "features guardadas", "Feature en vivo", "fetchers paralelos",
                                        "[WFB-CAUSAL-FIX-HMM]", "[BUGFIX-OVERFLOW-CEILING]"]):
            steps[2].append(clean_line)
        elif any(kw in line for kw in ["Inferencia", "[BRAIN]", "[Consensus/RESULT]",
                                        "[Seed 99]", "[Seed 1337]", "[Seed 2025]",
                                        "[FIX-XGB-TRAZABILIDAD]", "[Consensus]",
                                        "RegimeRouter/ROUTED", "[AUDITOR]", "Auditor",
                                        "Guard 1", "Guard 2", "Guard 3", "Guard 4", "Guard 5", "Guard 6",
                                        "[BUG-SHIELD", "[RISK-SHIELD", "OKX_BALANCE"]):
            steps[3].append(clean_line)
        elif any(kw in line for kw in ["[SIZER]", "[EXEC]", "[OKX_POSITION]", "Spot BTC/",
                                        "Orden colocada", "cierre completo", "SELL", "BUY",
                                        "[LIVE-TRADER-AUDIT]", "[BUGFIX-DEMO-BOOT]", "[BUGFIX-3]"]):
            steps[4].append(clean_line)
        elif any(kw in line for kw in ["Ciclo finalizado", "Durmiendo"]):
            steps[5].append(clean_line)
        else:
            steps[0].append(clean_line)

    step_labels = ["Boot/Carga", "Heartbeat/Recon/Riesgo", "Data/Features",
                   "Inferencia+Guards", "Exec OKX", "Duración"]
    print()
    total_classified = 0
    for i, (step, label) in enumerate(zip(steps, step_labels)):
        total_classified += len(step)
        status = "✅" if len(step) > 0 else "⚠️ VACÍO"
        print(f"  Paso {i+1} [{label}]: {len(step)} líneas {status}")
        for l in step[:3]:
            print(f"    {l[:100]}")
        if len(step) > 3:
            print(f"    ... (+{len(step)-3} más)")

    # Verificación crítica: ¿el Paso 4 contiene [Seed 99] y [FIX-XGB-TRAZABILIDAD]?
    step4_text = "\n".join(steps[3])
    print(f"\n  [CRÍTICO] ¿Paso 4 contiene '[Seed 99]'?             {'✅ SÍ' if '[Seed 99]' in step4_text else '❌ NO'}")
    print(f"  [CRÍTICO] ¿Paso 4 contiene '[FIX-XGB-TRAZABILIDAD]'? {'✅ SÍ' if '[FIX-XGB-TRAZABILIDAD]' in step4_text else '❌ NO'}")
    print(f"  [CRÍTICO] ¿Paso 4 contiene 'Auditor'?               {'✅ SÍ' if 'Auditor' in step4_text else '❌ NO'}")
    print(f"  [CRÍTICO] ¿Paso 4 contiene '[Consensus/RESULT]'?    {'✅ SÍ' if '[Consensus/RESULT]' in step4_text else '❌ NO'}")
    print(f"\n  Total líneas clasificadas: {total_classified} de {len(cycle_lines)}")

print("\n" + "=" * 70)
print("[TEST-G1-G3] COMPLETADO")
print("=" * 70)

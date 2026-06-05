"""
verify_audit_completeness.py
Revision cruzada sistematica: auditoria_wfb_run_20260530.md vs estado real del codigo
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config.settings import cfg
import pathlib

src_oos = pathlib.Path("luna/models/predict_oos.py").read_text(encoding="utf-8")
src_sf  = pathlib.Path("luna/models/signal_filter.py").read_text(encoding="utf-8")

OK   = "OK  "
FAIL = "FAIL"
PEND = "PEND"
NOTE = "NOTE"

rows = []

# ── HIPOTESIS ─────────────────────────────────────────────────────────────────
rows.append(("", "HIPOTESIS", ""))
rows.append((OK,   "H1 Gate BULL_TREND_WEAK",       "DESCARTADA — correcto no implementar"))
rows.append((OK,   "H5 OOD_KL_Distance gate",        "DESCARTADA — correcto no implementar"))
rows.append((NOTE, "H6 tribe_mult gate",             "TENDENCIA (p=0.116) — pendiente proxima run"))

# ── H2 ────────────────────────────────────────────────────────────────────────
rows.append(("", "H2 — Hard skip BEAR_CRASH (P0)", ""))
h2_ok = "H2-FIX" in src_oos and "BEAR_CRASH" in src_oos and "continue" in src_oos
rows.append((OK if h2_ok else FAIL, "Hard skip en predict_oos.py", "trades con kelly=0 ya no se logean"))

# ── H3 ────────────────────────────────────────────────────────────────────────
rows.append(("", "H3 — MetaLabeler threshold dinamico (P0+P1)", ""))
rows.append((OK if "H3-FIX" in src_sf else FAIL,
             "Codigo H3-FIX en signal_filter.py", ""))
rows.append((OK if "meta_v2_thresh_bull_strong" in src_sf else FAIL,
             "Lee thresh_bull_strong de settings (no hardcoded)", "era 0.50 hardcodeado"))
rows.append((OK if "meta_v2_rolling_percentile" in src_sf else FAIL,
             "CAPA-5 rolling percentile implementado", ""))
rows.append((OK if "CAPA-5" in src_sf else FAIL,
             "CAPA-5 en signal_filter.py", "nuevo mecanismo causal"))

thresh_strong   = getattr(cfg.metalabeler, "meta_v2_thresh_bull_strong",   None)
thresh_unstable = getattr(cfg.metalabeler, "meta_v2_thresh_bull_unstable", None)
rolling_pct     = getattr(cfg.metalabeler, "meta_v2_rolling_percentile",   None)
sim_online      = getattr(cfg.metalabeler, "simulate_online_recalibration", None)
min_prob        = getattr(cfg.metalabeler, "meta_v2_min_prob",              None)

rows.append((OK if thresh_strong   == 0.48  else FAIL, f"meta_v2_thresh_bull_strong = {thresh_strong}",   "derivado de Optuna, no OOS"))
rows.append((OK if thresh_unstable == 0.57  else FAIL, f"meta_v2_thresh_bull_unstable = {thresh_unstable}", "derivado de Optuna +5%"))
rows.append((OK if rolling_pct     == 0.60  else FAIL, f"meta_v2_rolling_percentile = {rolling_pct}",     "plateau q=0.60, no cherry-picked"))
rows.append((OK if sim_online      == False else FAIL, f"simulate_online_recalibration = {sim_online}",   "CAPA-4 off para no duplicar"))
rows.append((NOTE, f"meta_v2_min_prob = {min_prob} (piso, no threshold efectivo)",
             "El threshold real es dinamico via PROPUESTA-C + CAPA-5"))
rows.append((PEND, "Recalibracion Platt Scaling del meta-modelo (P1)",
             "Requiere reentrenamiento — para proxima iteracion"))

# ── H4 ────────────────────────────────────────────────────────────────────────
rows.append(("", "H4 — Consenso multi-seed (P0)", ""))
consensus = getattr(cfg.wfb, "ensemble_consensus_threshold", None)
embargo   = getattr(cfg.wfb, "soft_embargo_enabled", None)
rows.append((OK if consensus == 4  else FAIL, f"ensemble_consensus_CUTOFF = {consensus}", "era 2"))
rows.append((OK if embargo   == True else NOTE, f"soft_embargo_enabled = {embargo}", "24H embargo para consenso alto"))

# ── AJUSTES WFB ───────────────────────────────────────────────────────────────
rows.append(("", "AJUSTES WFB (solicitados en sesion)", ""))
max_pbo    = getattr(cfg.gauntlet, "max_pbo",              None)
max_seeds  = getattr(cfg.wfb,      "max_seeds_to_explore", None)
min_seeds  = getattr(cfg.wfb,      "min_seeds_to_approve", None)
prune_thr  = getattr(cfg.wfb,      "prune_threshold",      None)

rows.append((OK if max_pbo   == 0.45 else FAIL, f"max_pbo = {max_pbo}",   "era 0.22"))
rows.append((OK if max_seeds == 12   else FAIL, f"max_seeds_to_explore = {max_seeds}", "era 20"))
rows.append((OK if min_seeds == 3    else FAIL, f"min_seeds_to_approve = {min_seeds}", "era 5"))
rows.append((OK if prune_thr == 0.80 else FAIL, f"prune_CUTOFF = {prune_thr}",      "nuevo"))

# ── PENDIENTES ─────────────────────────────────────────────────────────────────
rows.append(("", "PENDIENTES (no implementados aun)", ""))
rows.append((PEND, "Analisis especifico W5 2026-Q1 (P2)", "causa del colapso WR=36.8%"))
rows.append((PEND, "Gate KL_distance inverso (P3)",        "H5 marginal p=0.116 — confirmar proxima run"))
rows.append((NOTE, "H6 tribe_mult gate",                   "p=0.116 — acumular N en proxima run"))

# ── PROTECCION SETTINGS ────────────────────────────────────────────────────────
rows.append(("", "PROTECCION SETTINGS (SOP restore)", ""))
import glob
backups = glob.glob("config/settings_backup_wfb_*.yaml")
orphans = [b for b in backups if "ORPHANED" in b]
actives = [b for b in backups if "ORPHANED" not in b]
rows.append((OK if not actives else FAIL,
             f"Backups activos sin ORPHANED: {len(actives)}",
             "0 = seguro relanzar"))
rows.append((OK, f"Backup antiguo neutralizado: {len(orphans)} ORPHANED", ""))

# ── IMPRIMIR ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("REVISION CRUZADA: auditoria_wfb_run_20260530.md vs CODIGO/SETTINGS")
print("=" * 70)

for status, item, note in rows:
    if status == "":
        print(f"\n  ── {item} ──")
        continue
    marker = {"OK  ": "[OK  ]", "FAIL": "[FAIL]", "PEND": "[PEND]", "NOTE": "[NOTE]"}[status]
    note_str = f"  <- {note}" if note else ""
    print(f"  {marker} {item}{note_str}")

# Conteo final
ok_count   = sum(1 for s, _, _ in rows if s == OK)
fail_count = sum(1 for s, _, _ in rows if s == FAIL)
pend_count = sum(1 for s, _, _ in rows if s == PEND)
note_count = sum(1 for s, _, _ in rows if s == NOTE)

print()
print("=" * 70)
print(f"RESUMEN: {ok_count} OK | {fail_count} FAIL | {pend_count} PENDIENTES | {note_count} NOTAS")
if fail_count == 0:
    print("ESTADO: LISTO PARA RELANZAR RUN")
else:
    print("ESTADO: REVISAR FALLOS ANTES DE RELANZAR")
print("=" * 70)

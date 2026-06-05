"""
fix_crit01_nsplits_adaptive.py
Aplica FIX-CRIT-01: n_splits_is adaptativo al tamaño del dataset del AGENTE.
Con n_train=766 (bear), _n_months=0 -> n_splits=3 -> 255 events/fold (T suficiente).
El bug era que se calculaba n_months sobre el calendario completo (6-month buckets)
en lugar de basarse en el tamaño real del agente filtrado por regimen.
"""
import ast, sys

fp = r'g:\Mi unidad\ia\luna_v2\luna\models\train_xgboost_v2.py'

with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

if 'FIX-CRIT-01-NSPLITS' in content:
    print("FIX-CRIT-01 ya aplicado -- abortando")
    sys.exit(0)

# Patron exacto (ya tiene CRIT-02 aplicado, no afecta esta region)
OLD = (
    "        _n_months = max(1, len(self.X) // (24 * 30 * 6))\n"
    "        _n_splits_is = max(3, min(6, _n_months))\n"
    "        _tscv = TimeSeriesSplit(n_splits=_n_splits_is, gap=_embargo_gap)\n"
)

if OLD not in content:
    # Intentar con \r\n
    OLD_CRLF = OLD.replace('\n', '\r\n')
    if OLD_CRLF in content:
        OLD = OLD_CRLF
        print("Patron encontrado con \\r\\n")
    else:
        print("ERROR: patron n_months/_n_splits_is no encontrado")
        # Mostrar contexto para debug
        idx = content.find('_n_months = max(1')
        if idx >= 0:
            print(f"Encontrado a idx {idx}: {repr(content[idx:idx+150])}")
        sys.exit(1)

eol = '\r\n' if '\r\n' in OLD else '\n'

NEW = (
    "        # [FIX-CRIT-01-NSPLITS 2026-05-30] n_splits adaptativo al tamanio del agente." + eol
    + "        # BUG: _n_months = len(X)//4320 = 766//4320 = 0 -> n_splits=3 pero era coincidencia correcta." + eol
    + "        # El problema real: con n=766 y n_splits=3, cada fold_test tiene ~255 filas" + eol
    + "        # PERO el DSR con T=255 y n_trials=1-5 puede ser cercano a 0 si los trials son pocos." + eol
    + "        # Fix: calcular n_splits explicitamente por tamanio del agente, no por meses calendario." + eol
    + "        # Regla: n<2000 -> 3 splits (T_test~33%), n<5000 -> 5 splits, n>=5000 -> 6 splits." + eol
    + "        _n_train_for_splits = len(self.X)" + eol
    + "        if _n_train_for_splits < 2000:" + eol
    + "            _n_splits_is = 3" + eol
    + "        elif _n_train_for_splits < 5000:" + eol
    + "            _n_splits_is = 5" + eol
    + "        else:" + eol
    + "            _n_splits_is = 6" + eol
    + "        if trial.number == 0:" + eol
    + "            print(  # RULE[fixbugsprints.md]" + eol
    + "                f\"[FIX-CRIT-01-NSPLITS] n_splits_is={_n_splits_is} para n_train={_n_train_for_splits}\"" + eol
    + "                f\" | embargo_gap={_embargo_gap}H | T_test~{_n_train_for_splits // (_n_splits_is + 1)} eventos\"" + eol
    + "            )" + eol
    + "            logger.info(" + eol
    + "                \"[FIX-CRIT-01-NSPLITS] Optuna TimeSeriesSplit: n_splits=%d | n_train=%d | gap=%dH\"," + eol
    + "                _n_splits_is, _n_train_for_splits, _embargo_gap" + eol
    + "            )" + eol
    + "        _tscv = TimeSeriesSplit(n_splits=_n_splits_is, gap=_embargo_gap)" + eol
)

content = content.replace(OLD, NEW, 1)

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

try:
    ast.parse(content)
    print("SYNTAX OK: train_xgboost_v2.py")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

print("FIX-CRIT-01-NSPLITS aplicado correctamente")

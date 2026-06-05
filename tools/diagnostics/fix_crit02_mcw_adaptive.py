"""
fix_crit02_mcw_adaptive.py
Aplica FIX-CRIT-02: min_child_weight_max adaptativo al tamaño del dataset.
Previene model collapse cuando n_train es pequeño (agente BEAR: n~766).
"""
import ast, sys

fp = r'g:\Mi unidad\ia\luna_v2\luna\models\train_xgboost_v2.py'

with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

if 'FIX-CRIT-02-MCW-ADAPTIVE' in content:
    print("FIX-CRIT-02 ya aplicado -- abortando")
    sys.exit(0)

# El patron exacto a buscar
OLD = (
    "        _gamma_max = sp.gamma_max\n"
    "        _mcw_min   = sp.min_child_weight_min\n"
    "        _mcw_max   = sp.min_child_weight_max\n"
    "        \n"
    "        if trial.number == 0:\n"
    "            print(f\"[LUNA-V2-REGULARIZATION] STRICT BOUNDS active | min_child_weight range=[{_mcw_min}, {_mcw_max}], max_depth in [{sp.max_depth_min}, {sp.max_depth_max}], gamma_max={_gamma_max}\")\n"
)

if OLD not in content:
    print("ERROR: patron LUNA-V2-REGULARIZATION no encontrado en train_xgboost_v2.py")
    # Intentar con \r\n
    OLD2 = OLD.replace('\n', '\r\n')
    if OLD2 in content:
        print("Encontrado con \\r\\n, ajustando...")
        OLD = OLD2
    else:
        sys.exit(1)

NEW = (
    "        _gamma_max = sp.gamma_max\n"
    "        _mcw_min   = sp.min_child_weight_min\n"
    "        _mcw_max   = sp.min_child_weight_max\n\n"
    "        # [FIX-CRIT-02-MCW-ADAPTIVE 2026-05-30] Adaptar min_child_weight_max al n_train del AGENTE.\n"
    "        # Con n_train=766 (bear) y min_child_weight_max=100, ningun arbol puede crecer:\n"
    "        # depth=4 -> 766/16=47 samples/hoja < MCW=100 -> todos los arboles son triviales.\n"
    "        # El floor: MCW_max = min(MCW_max_settings, max(10, n_train // 20))\n"
    "        # Con n=766: min(100, max(10, 38)) = 38 -> depth=4: 766/16=47 >= 38 -> arboles crecen.\n"
    "        _n_train_agent = len(self.X)\n"
    "        _mcw_max_adaptive = min(_mcw_max, max(10, _n_train_agent // 20))\n"
    "        if _mcw_max_adaptive < _mcw_max:\n"
    "            print(  # RULE[fixbugsprints.md]\n"
    "                f\"[FIX-CRIT-02-MCW-ADAPTIVE] min_child_weight_max reducido: {_mcw_max} -> {_mcw_max_adaptive}\"\n"
    "                f\" (n_train={_n_train_agent}, n_train//20={_n_train_agent//20}) -- previene model collapse\"\n"
    "            )\n"
    "            logger.warning(\n"
    "                \"[FIX-CRIT-02-MCW-ADAPTIVE] n_train=%d pequeno -> min_child_weight_max=%d (era %d) \"\n"
    "                \"-- prevencion de model collapse activa\",\n"
    "                _n_train_agent, _mcw_max_adaptive, _mcw_max\n"
    "            )\n"
    "            _mcw_max = _mcw_max_adaptive\n"
    "        \n"
    "        if trial.number == 0:\n"
    "            print(f\"[LUNA-V2-REGULARIZATION] STRICT BOUNDS active | min_child_weight range=[{_mcw_min}, {_mcw_max}], max_depth in [{sp.max_depth_min}, {sp.max_depth_max}], gamma_max={_gamma_max}\")\n"
)

content = content.replace(OLD, NEW, 1)

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

# Verificar
try:
    ast.parse(content)
    print("SYNTAX OK: train_xgboost_v2.py")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

print("FIX-CRIT-02-MCW-ADAPTIVE aplicado correctamente")

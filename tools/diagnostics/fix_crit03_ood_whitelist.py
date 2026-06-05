"""
fix_crit03_ood_whitelist.py  v2
Aplica FIX-CRIT-03 de forma robusta usando manipulacion de strings.
"""
import ast, sys

fp = r'g:\Mi unidad\ia\luna_v2\luna\utils\ood_feature_guard.py'

# Leer original
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

# Verificar que no está ya aplicado
if 'FIX-CRIT-03' in content:
    print("FIX-CRIT-03 ya aplicado — abortando")
    sys.exit(0)

# --- CAMBIO 1: Añadir self.structural_features en __init__ ---
OLD_INIT = "        self.block_on_low_unique = block_on_low_unique\n\n        # Intentar leer thresholds"
NEW_INIT = (
    "        self.block_on_low_unique = block_on_low_unique\n"
    "        self.structural_features: list = []  # [FIX-CRIT-03] features exentas del OOD Guard\n\n"
    "        # Intentar leer thresholds"
)
assert OLD_INIT in content, "CAMBIO 1: patron no encontrado"
content = content.replace(OLD_INIT, NEW_INIT, 1)
print("[1] __init__ structural_features OK")

# --- CAMBIO 2: Añadir carga desde settings en _load_from_settings ---
OLD_LOAD = (
    '                logger.debug("[OOSFeatureGuard] Thresholds cargados desde settings.yaml [ood_guard]")\n'
)
NEW_LOAD = (
    '                # [FIX-CRIT-03] Cargar whitelist de features estructurales (exentas del bloqueo CONSTANT)\n'
    '                self.structural_features = list(getattr(_ood, "structural_features", []))\n'
    '                if self.structural_features:\n'
    '                    print(f"[FIX-CRIT-03] OOD Guard structural_features exentas: {self.structural_features}")  # RULE[fixbugsprints.md]\n'
    '                    logger.info(\n'
    '                        "[FIX-CRIT-03] OOD Guard: %d features estructurales exentas de bloqueo CONSTANT/LOW_STD: %s",\n'
    '                        len(self.structural_features), self.structural_features\n'
    '                    )\n'
    '                logger.debug("[OOSFeatureGuard] Thresholds cargados desde settings.yaml [ood_guard]")\n'
)
assert OLD_LOAD in content, "CAMBIO 2: patron no encontrado"
content = content.replace(OLD_LOAD, NEW_LOAD, 1)
print("[2] _load_from_settings structural_features OK")

# --- CAMBIO 3: Insertar check exencion en _analyze_feature ---
OLD_CRIT1 = (
    "        # Criterio 1: CONSTANT — demasiado % con el mismo valor en OOS\n"
    "        if oos_constant_pct >= self.max_constant_pct:\n"
)
NEW_CRIT1 = (
    "        # [FIX-CRIT-03] Exencion para features estructurales (e.g. HMM_Regime).\n"
    "        # HMM_Regime constante en OOS es informacion legitima del regimen actual,\n"
    "        # no degeneracion. El OOD Guard no debe eliminar features estructurales.\n"
    "        if feat in getattr(self, 'structural_features', []):\n"
    "            print(  # RULE[fixbugsprints.md]\n"
    '                f"[FIX-CRIT-03] OOD Guard: feature estructural \'{feat}\' exenta "\n'
    '                f"(oos_unique={oos_nunique}, oos_const={oos_constant_pct:.1%}) -- STRUCTURAL_EXEMPT"\n'
    "            )\n"
    "            logger.info(\n"
    '                "[FIX-CRIT-03] OOD Guard: \'%s\' exenta -- structural feature "\n'
    '                "(oos_unique=%d, oos_const=%.1f%%)", feat, oos_nunique, oos_constant_pct * 100\n'
    "            )\n"
    "            return OODFeatureReport(\n"
    "                feature=feat, blocked=False, reason='STRUCTURAL_EXEMPT', severity='OK',\n"
    "                train_std=train_std, train_nunique=train_nunique,\n"
    "                oos_std=oos_std, oos_nunique=oos_nunique,\n"
    "                oos_constant_pct=oos_constant_pct, std_ratio=std_ratio,\n"
    "            )\n\n"
    "        # Criterio 1: CONSTANT — demasiado % con el mismo valor en OOS\n"
    "        if oos_constant_pct >= self.max_constant_pct:\n"
)
assert OLD_CRIT1 in content, "CAMBIO 3: patron no encontrado"
content = content.replace(OLD_CRIT1, NEW_CRIT1, 1)
print("[3] _analyze_feature structural_exempt OK")

# Escribir resultado
with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

# Verificar sintaxis
try:
    ast.parse(content)
    print("SYNTAX OK: ood_feature_guard.py")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

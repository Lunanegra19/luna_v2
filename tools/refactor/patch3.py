import sys
import re

with open('luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_idea_a = False
for i, line in enumerate(lines):
    if 'IDEA-A: calcular base_rate IS' in line:
        in_idea_a = True
        new_lines.append(line)
        continue
        
    if in_idea_a:
        if 'self._brier_adaptive_gate = _brier_adaptive_gate' in line:
            in_idea_a = False
            # Append our logic instead
            new_lines.append('''        self._base_rate_is = float(self.y.mean()) if self.y is not None and len(self.y) > 0 else 0.50
        if self._base_rate_is <= 0.01 or self._base_rate_is >= 0.99:
            _brier_adaptive_gate = None  # NO_OPERABLE
            logger.warning(
                "[IDEA-A][FIX-IDEA-A-01] base_rate_IS=%.3f -> agente degenerado (sin muestras del regimen). "
                "brier_adaptive_gate=None (NO_OPERABLE). No se calculara gate de calibracion.",
                self._base_rate_is
            )
        else:
            try:
                _naive_is_fold = self.study.best_trial.user_attrs.get("naive_is")
            except Exception:
                _naive_is_fold = None
            
            if _naive_is_fold is not None:
                _brier_naive_true = _naive_is_fold
                logger.info(f"[FIX-IDEA-A-01] Usando Naive Brier ({_brier_naive_true:.4f}) alineado a los folds de Optuna, "
                            f"en lugar del global ({self._base_rate_is * (1 - self._base_rate_is):.4f})")
            else:
                _brier_naive_true = self._base_rate_is * (1 - self._base_rate_is)

            _brier_adaptive_gate = round(_brier_naive_true + 0.010, 4)
            logger.info("[IDEA-A] base_rate_IS=%.3f brier_naive_true=%.4f brier_adaptive_gate=%.4f",
                        self._base_rate_is,
                        _brier_naive_true,
                        _brier_adaptive_gate)
        self._brier_adaptive_gate = _brier_adaptive_gate
''')
        continue
    
    new_lines.append(line)

with open('luna/models/train_xgboost_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

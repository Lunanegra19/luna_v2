import re

path = 'd:/Andres/luna_v2/luna/pipeline_integrity.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the win rate block
old_wr_block = r'''        if wr < 20:
            print\(
                f"\{_TAG\}   \[\{window_id\}\] .*? CRITICAL: WR=\{wr:\.1f\}% < 20%\. "
                f"El modelo est.*? prediciendo PEOR que azar\. Posibles causas: "
                f"se.*?al invertida, calibrador OOB, FIX-REG-01 activo\."
            \)
            logger\.critical\(f"\{_TAG\} \[\{window_id\}\] WR=\{wr:\.1f\}% .*? peor que azar\."\)
        elif wr < 35:
            print\(f"\{_TAG\}   \[\{window_id\}\] .*? WARNING: WR=\{wr:\.1f\}% .*? bajo \(azar=50%\)\. Verificar pipeline\."\)
            logger\.warning\(f"\{_TAG\} \[\{window_id\}\] WR bajo: \{wr:\.1f\}%"\)
        elif wr > 80:
            print\(
                f"\{_TAG\}   \[\{window_id\}\] .*? WARNING: WR=\{wr:\.1f\}% > 80% .*? sospechosamente alto\. "
                f"Verificar look-ahead bias, embargo, y PurgedKFold\."
            \)
            logger\.warning\(f"\{_TAG\} \[\{window_id\}\] WR sospechosamente alto: \{wr:\.1f\}%"\)
        else:
            print\(f"\{_TAG\}   \[\{window_id\}\] .*?\. WR=\{wr:\.1f\}% dentro del rango esperado\."\)'''

new_wr_block = '''        if n < 30:
            msg = f"{_TAG}   [{window_id}] INFO: WR={wr:.1f}% sobre {n} trades (Insignificante segun SOP R8)."
            print(msg)
            logger.info(msg)
        else:
            if wr < 20:
                print(
                    f"{_TAG}   [{window_id}] CRITICAL: WR={wr:.1f}% < 20%. "
                    f"El modelo esta prediciendo PEOR que azar. Posibles causas: "
                    f"senal invertida, calibrador OOB, FIX-REG-01 activo."
                )
                logger.critical(f"{_TAG} [{window_id}] WR={wr:.1f}% - peor que azar.")
            elif wr < 35:
                print(f"{_TAG}   [{window_id}] WARNING: WR={wr:.1f}% - bajo (azar=50%). Verificar pipeline.")
                logger.warning(f"{_TAG} [{window_id}] WR bajo: {wr:.1f}%")
            elif wr > 80:
                print(
                    f"{_TAG}   [{window_id}] WARNING: WR={wr:.1f}% > 80% - sospechosamente alto. "
                    f"Verificar look-ahead bias, embargo, y PurgedKFold."
                )
                logger.warning(f"{_TAG} [{window_id}] WR sospechosamente alto: {wr:.1f}%")
            else:
                print(f"{_TAG}   [{window_id}] OK. WR={wr:.1f}% dentro del rango esperado.")'''

if 'if n < 30:' not in content:
    content = re.sub(old_wr_block, new_wr_block, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch script finished.")

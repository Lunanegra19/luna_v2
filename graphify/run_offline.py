"""
graphify/run_offline.py
======================
Script de diagnóstico y mapeo de arquitectura offline para Luna V2.
Ejecuta la extracción AST completa del repositorio, detecta comunidades de
archivos y genera reportes e interactivos HTML en la carpeta local graphify/out/.
"""

import os
import sys
import json
from pathlib import Path

# Configurar UTF-8 para consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    print("=" * 80)
    print("   INICIANDO MAPEO DE ARQUITECTURA OFFLINE EN /graphify (LUNA V2)")
    print("=" * 80)
    
    # 1. Crear carpeta de salida encapsulada en la raíz
    out_dir = _ROOT / "graphify" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Carpeta de salida garantizada: {out_dir}")

    # Importar librerías de graphify
    try:
        print("[INFO] Cargando librerías de Graphify...")
        from graphify.detect import detect
        from graphify.extract import collect_files, extract
        from graphify.build import build_from_json
        from graphify.cluster import cluster, score_all
        from graphify.analyze import god_nodes, surprising_connections, suggest_questions
        from graphify.report import generate
        from graphify.export import to_json, to_html
        print("[INFO] Librerías de Graphify cargadas correctamente.")
    except ImportError as e:
        print(f"[CRITICAL] Error al importar graphifyy: {e}")
        print("Asegúrate de que 'graphifyy' está instalado en el entorno de Python.")
        sys.exit(1)

    # 2. Detección rápida y optimizada para Google Drive (Filtro por Carpetas Críticas)
    print("\n[FASE 1] Detectando archivos (Optimizado para Google Drive)...")
    
    # Carpetas que queremos escanear
    TARGET_FOLDERS = ["luna", "scripts", "tools", "config", "docs"]
    
    code_files_str = []
    doc_files_str = []
    
    for folder_name in TARGET_FOLDERS:
        folder_path = _ROOT / folder_name
        if not folder_path.exists():
            continue
            
        # Buscar recursivamente
        for p in folder_path.rglob("*"):
            if not p.is_file():
                continue
            # Ignorar caches y directorios de graphify
            if "__pycache__" in p.parts or ".pytest_cache" in p.parts or "graphify" in p.parts or "graphify-out" in p.parts:
                continue
                
            # Clasificar por extensión
            if p.suffix == ".py":
                code_files_str.append(str(p))
            elif p.suffix in [".md", ".txt", ".yaml", ".yml"]:
                doc_files_str.append(str(p))
                
    detection_res = {
        "total_files": len(code_files_str) + len(doc_files_str),
        "total_words": len(code_files_str) * 300 + len(doc_files_str) * 1000, # estimación aproximada para reportes
        "files": {
            "code": code_files_str,
            "docs": doc_files_str
        }
    }
    
    # Escribir resultado de detección
    detect_json_path = out_dir / ".graphify_detect.json"
    detect_json_path.write_text(json.dumps(detection_res, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Mostrar resumen
    print(f"[OK] Detección completada en disco:")
    print(f"  - código (.py): {len(code_files_str)} archivos")
    print(f"  - documentación/config (.md, .yaml, .txt): {len(doc_files_str)} archivos")

    # 3. Extracción de estructura AST (Código Python)
    print("\n[FASE 2] Iniciando extracción estructural AST de archivos de código...")
    code_files = [Path(f) for f in code_files_str]
            
    print(f"[INFO] Procesando {len(code_files)} archivos de código fuente Python...")
    
    if code_files:
        try:
            ast_res = extract(code_files)
            print(f"[OK] Extracción AST completada exitosamente.")
            print(f"  - Nodos encontrados: {len(ast_res['nodes'])}")
            print(f"  - Relaciones (Edges) encontradas: {len(ast_res['edges'])}")
        except Exception as e:
            print(f"[ERROR] Fallo en la extracción AST paralela: {e}")
            print("[INFO] Reintentando de forma secuencial...")
            # Fallback secuencial manual si falla la extracción paralela en Windows
            ast_res = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}
            from graphify.extract import extract_single_file
            for cf in code_files:
                try:
                    single_res = extract_single_file(cf)
                    ast_res["nodes"].extend(single_res.get("nodes", []))
                    ast_res["edges"].extend(single_res.get("edges", []))
                except Exception as ex:
                    print(f"  [WARN] No se pudo extraer {cf.name}: {ex}")
            print(f"[OK] Extracción AST secuencial completada. Nodos: {len(ast_res['nodes'])}, Edges: {len(ast_res['edges'])}")
    else:
        print("[WARN] No se detectaron archivos de código en el corpus.")
        ast_res = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}

    # Escribir archivos temporales de extracción
    ast_json_path = out_dir / ".graphify_ast.json"
    ast_json_path.write_text(json.dumps(ast_res, indent=2, ensure_ascii=False), encoding="utf-8")
    
    extract_json_path = out_dir / ".graphify_extract.json"
    extract_json_path.write_text(json.dumps(ast_res, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Construcción del Grafo y Detección de Comunidades (Clustering)
    print("\n[FASE 3] Construyendo el Grafo de conocimiento y agrupando comunidades...")
    G = build_from_json(ast_res)
    
    if G.number_of_nodes() == 0:
        print("[CRITICAL] El Grafo está vacío. La extracción no produjo ningún nodo de código.")
        sys.exit(1)
        
    communities = cluster(G)
    cohesion = score_all(G, communities)
    
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    
    # Crear etiquetas para comunidades
    labels = {}
    for cid, nodes in communities.items():
        community_nodes = [G.nodes[n] for n in nodes if n in G.nodes]
        names = [n.get("label", "") for n in community_nodes]
        
        # Etiquetado heurístico basado en palabras clave
        lbl = f"Grupo {cid}"
        names_str = " ".join(names).lower()
        if any(w in names_str for w in ["ensemble", "voting", "inf", "pred"]):
            lbl = "Live Ensemble Inference"
        elif any(w in names_str for w in ["okx", "broker", "order", "position"]):
            lbl = "OKX Connector & Exec"
        elif any(w in names_str for w in ["wfb", "backtest", "walk", "prun"]):
            lbl = "Walk-Forward Engine"
        elif any(w in names_str for w in ["optimal", "seed", "krogh", "vedelsby"]):
            lbl = "Seed Optimization"
        elif any(w in names_str for w in ["settings", "config", "yaml"]):
            lbl = "System Configuration"
        elif any(w in names_str for w in ["feature", "sfi", "parquet", "ingest"]):
            lbl = "Data & Feature Pipelines"
        elif any(w in names_str for w in ["test", "mock", "assert"]):
            lbl = "Unit Validation Tests"
        elif any(w in names_str for w in ["audit", "forensic", "check", "diag"]):
            lbl = "Diagnostics & Audit Tools"
        
        labels[cid] = lbl
        print(f"  - Comunidad {cid}: {lbl} ({len(nodes)} nodos, Cohesión: {cohesion.get(cid, 0.0):.2f})")

    # 5. Guardar metadatos y análisis
    analysis = {
        "communities": {str(k): v for k, v in communities.items()},
        "cohesion": {str(k): v for k, v in cohesion.items()},
        "gods": gods,
        "surprises": surprises,
    }
    analysis_json_path = out_dir / ".graphify_analysis.json"
    analysis_json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    
    labels_json_path = out_dir / ".graphify_labels.json"
    labels_json_path.write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding="utf-8")

    # 6. Generar reportes finales y archivos de visualización
    print("\n[FASE 4] Exportando visualización HTML interactiva y reportes...")
    
    # Sugerir preguntas dinámicas
    questions = suggest_questions(G, communities, labels)
    
    # Generar reporte escrito
    tokens = {"input": 0, "output": 0}
    report = generate(G, communities, cohesion, labels, gods, surprises, detection_res, tokens, str(_ROOT), suggested_questions=questions)
    
    report_md_path = out_dir / "GRAPH_REPORT.md"
    report_md_path.write_text(report, encoding="utf-8")
    
    # Exportar JSON y HTML
    to_json(G, communities, str(out_dir / "graph.json"), force=True)
    to_html(G, communities, str(out_dir / "graph.html"), community_labels=labels)
    
    print("\n" + "=" * 80)
    print("   PROCESAMIENTO DE MAPEO DE ARQUITECTURA COMPLETADO CON ÉXITO")
    print("=" * 80)
    print(f"[OK] Grafo persistido en: {out_dir / 'graph.json'}")
    print(f"[OK] Visualización interactiva en: {out_dir / 'graph.html'}")
    print(f"[OK] Reporte estructural en: {out_dir / 'GRAPH_REPORT.md'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

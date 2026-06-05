import os
import sys

def main():
    archive_dir = "g:\\Mi unidad\\ia\\Luna v1\\data\\archive"
    print(f"[CLEANUP-V1-START] Iniciando limpieza segura de matrices de entrenamiento en: {archive_dir}")
    sys.stdout.flush()
    
    if not os.path.exists(archive_dir):
        print(f"[CLEANUP-V1-ERROR] La ruta del archivo no existe: {archive_dir}")
        return
        
    total_files_deleted = 0
    total_bytes_reclaimed = 0
    
    # Recorremos la carpeta archive de Luna V1
    for dirpath, dirnames, filenames in os.walk(archive_dir):
        for f in filenames:
            # Filtro estricto: Debe comenzar con 'features_train' y terminar con '.parquet'
            if f.startswith("features_train") and f.endswith(".parquet"):
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        size = os.path.getsize(fp)
                        os.remove(fp)
                        total_files_deleted += 1
                        total_bytes_reclaimed += size
                        
                        # Cada 100 archivos borrados imprimimos progreso
                        if total_files_deleted % 100 == 0:
                            mb_reclaimed = round(total_bytes_reclaimed / (1024 * 1024), 2)
                            print(f"[CLEANUP-V1-PROGRESS] Borrados {total_files_deleted} archivos... {mb_reclaimed} MB liberados.")
                            sys.stdout.flush()
                    except OSError as e:
                        print(f"[CLEANUP-V1-WARN] No se pudo eliminar {fp}: {e}")
                        sys.stdout.flush()
                        
    total_mb_reclaimed = round(total_bytes_reclaimed / (1024 * 1024), 2)
    total_gb_reclaimed = round(total_bytes_reclaimed / (1024 * 1024 * 1024), 2)
    
    print("\n" + "=" * 80)
    print(f"      [CLEANUP-V1-SUCCESS] LIMPIEZA DE ARCHIVO V1 COMPLETADA CON EXITO!")
    print(f"      - Total de Archivos Parquet Eliminados: {total_files_deleted}")
    print(f"      - Espacio Total Reclamado: {total_mb_reclaimed} MB ({total_gb_reclaimed} GB)")
    print(f"      - Nota: Los reportes, modelos, validaciones y holdouts siguen intactos.")
    print("=" * 80 + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()

import os
import sys
import tarfile
import subprocess
import tempfile
from pathlib import Path

# Cargar ruta del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Intentar leer ORACLE_HOST desde .env
def get_oracle_host():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("ORACLE_HOST="):
                    return line.strip().split("=")[1].strip()
    return "178.105.197.191"

# Intentar localizar la clave SSH
def get_ssh_key():
    key_path = Path("C:/Users/Usuario/.ssh/id_ed25519")
    if key_path.exists():
        return str(key_path)
    # Fallback standard
    user_home = Path(os.path.expanduser("~"))
    fallback_key = user_home / ".ssh" / "id_ed25519"
    if fallback_key.exists():
        return str(fallback_key)
    return None

def main():
    print("[LUNA-SYNC] INICIANDO PROCESO DE SINCRONIZACION DE REPORTES DE BACKTEST")
    
    oracle_host = get_oracle_host()
    ssh_key = get_ssh_key()
    
    if not ssh_key:
        print("[LUNA-SYNC-ERROR] ERROR CRITICO: No se pudo localizar la clave SSH 'id_ed25519'.")
        sys.exit(1)
        
    print(f"[LUNA-SYNC] Host Destino: {oracle_host}")
    print(f"[LUNA-SYNC] Clave SSH: {ssh_key}")
    
    local_reports_dir = PROJECT_ROOT / "data" / "reports"
    if not local_reports_dir.exists():
        print(f"[LUNA-SYNC-ERROR] ERROR: Directorio local de reportes no existe: {local_reports_dir}")
        sys.exit(1)
        
    # Crear un archivo temporal tar.gz
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tmp_tar_path = Path(tmp_file.name)
        
    print(f"[LUNA-SYNC] Comprimiendo {local_reports_dir} en {tmp_tar_path}...")
    
    try:
        with tarfile.open(tmp_tar_path, "w:gz") as tar:
            # Añadir el directorio de reportes, guardando la ruta relativa reports/
            tar.add(local_reports_dir, arcname="reports")
        print(f"[LUNA-SYNC] Compresion completada con exito. Tamanio: {tmp_tar_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[LUNA-SYNC-ERROR] ERROR durante la compresion: {e}")
        if tmp_tar_path.exists():
            os.unlink(tmp_tar_path)
        sys.exit(1)
        
    # Paso 1: Asegurarse de que el directorio remoto /root/luna_v2/data/ existe
    print("[LUNA-SYNC] Asegurando directorio remoto en el servidor...")
    remote_mkdir_cmd = [
        "ssh", "-i", ssh_key, 
        "-o", "StrictHostKeyChecking=no",
        f"root@{oracle_host}", 
        "mkdir -p /root/luna_v2/data/"
    ]
    
    try:
        subprocess.run(remote_mkdir_cmd, capture_output=True, text=True, check=True)
        print("[LUNA-SYNC] Directorio remoto verificado/creado.")
    except subprocess.CalledProcessError as e:
        print(f"[LUNA-SYNC-ERROR] ERROR ejecutando mkdir remoto: {e.stderr}")
        if tmp_tar_path.exists():
            os.unlink(tmp_tar_path)
        sys.exit(1)
        
    # Paso 2: Subir el archivo tar.gz via scp
    print("[LUNA-SYNC] Subiendo archivo comprimido via SCP...")
    remote_tar_path = "/root/luna_v2/data/reports.tar.gz"
    scp_cmd = [
        "scp", "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        str(tmp_tar_path),
        f"root@{oracle_host}:{remote_tar_path}"
    ]
    
    try:
        subprocess.run(scp_cmd, check=True)
        print("[LUNA-SYNC] Archivo subido con exito.")
    except subprocess.CalledProcessError as e:
        print(f"[LUNA-SYNC-ERROR] ERROR subiendo archivo via SCP: {e}")
        if tmp_tar_path.exists():
            os.unlink(tmp_tar_path)
        sys.exit(1)
        
    # Paso 3: Descomprimir el archivo en el VPS y limpiar el tar.gz remoto
    print("[LUNA-SYNC] Descomprimiendo archivos en el servidor remoto...")
    remote_extract_cmd = [
        "ssh", "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        f"root@{oracle_host}",
        f"tar -xzf {remote_tar_path} -C /root/luna_v2/data/ && rm -f {remote_tar_path}"
    ]
    
    try:
        subprocess.run(remote_extract_cmd, check=True)
        print("[LUNA-SYNC] ¡SINCRONIZACION COMPLETADA CON EXITO!")
        print("[LUNA-SYNC] Todos los veredictos locales, historicos y graficos estan ahora en el VPS.")
    except subprocess.CalledProcessError as e:
        print(f"[LUNA-SYNC-ERROR] ERROR durante la descompresion remota: {e}")
        
    # Limpieza local
    if tmp_tar_path.exists():
        os.unlink(tmp_tar_path)
        print("[LUNA-SYNC] Limpieza local completada.")

if __name__ == "__main__":
    main()

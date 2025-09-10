import duckdb
import pandas as pd
import os
import glob

def build_duckdb():
    """Construye o actualiza base DuckDB desde todos los CSVs disponibles"""
    
    os.makedirs("data", exist_ok=True)
    duckdb_path = "data/scada_recovery.duckdb"
    
    # Buscar CSVs
    csv_files = sorted(set(glob.glob("data/*.csv") + glob.glob("*.csv")), key=os.path.getmtime)
    
    if not csv_files:
        print("No se encontraron archivos CSV")
        return False
    
    # Tags necesarios
    copper_tags = ['PowerBi.COU1CD2001CU', 'PowerBi.COU1RD001CU', 'PowerBi.COU1CF001CU', 'PowerBi.COU1-RCT-CU']
    
    # Conexión a DuckDB
    conn = duckdb.connect(duckdb_path)
    
    # Crear tabla si no existe
    conn.execute("""
        CREATE TABLE IF NOT EXISTS copper_data (
            ts_origin TIMESTAMP,
            tag_id VARCHAR,
            value_ts DOUBLE
        )
    """)
    
    # Obtener última fecha ya registrada
    last_ts = conn.execute("SELECT MAX(ts_origin) FROM copper_data").fetchone()[0]
    
    total_records = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_copper = df[df['tag_id'].isin(copper_tags)][['ts_origin', 'tag_id', 'value_ts']].copy()
            
            if len(df_copper) > 0:
                df_copper['ts_origin'] = pd.to_datetime(df_copper['ts_origin'])
                
                # Filtrar solo registros nuevos si ya hay datos
                if last_ts is not None:
                    df_copper = df_copper[df_copper['ts_origin'] > pd.to_datetime(last_ts)]
                
                if len(df_copper) > 0:
                    conn.execute("INSERT INTO copper_data SELECT * FROM df_copper")
                    total_records += len(df_copper)
                    print(f"Procesado: {os.path.basename(csv_file)} - {len(df_copper)} registros")
            
        except Exception as e:
            print(f"Error en {csv_file}: {e}")
    
    # Crear índice si no existe
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_tag ON copper_data(ts_origin, tag_id)")
    
    # Resumen
    summary = conn.execute("SELECT MIN(ts_origin) as inicio, MAX(ts_origin) as fin, COUNT(*) as total FROM copper_data").fetchone()
    conn.close()
    
    if total_records > 0:
        print(f"Base actualizada: {summary[2]:,} registros ({summary[0]} a {summary[1]})")
        return True
    else:
        print("No se agregaron registros nuevos")
        return True  # Devuelve True aunque no haya cambios para indicar que la DB está lista

if __name__ == "__main__":
    if build_duckdb():
        print("DuckDB listo para usar")
    else:
        print("Error en el proceso")

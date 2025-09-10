import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Importar DuckDB solo si está disponible
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

st.set_page_config(
    page_title="Panel de Control - Recuperación Cu | Atalaya Mining",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_and_process_data(file_path):
    """Carga y procesa datos desde CSV o Excel"""
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df['ts_origin'] = pd.to_datetime(df['ts_origin'])
        df['ts'] = pd.to_datetime(df['ts'])
        
        copper_tags = [
            'PowerBi.COU1CD2001CU',
            'PowerBi.COU1RD001CU',
            'PowerBi.COU1CF001CU',
            'PowerBi.COU1-RCT-CU'
        ]
        return df[df['tag_id'].isin(copper_tags)].copy()
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

@st.cache_data(ttl=900)  # Cache 15 minutos 
def load_from_duckdb(days_back=30, db_path="/app/data/scada_recovery.duckdb"):
    """Carga datos desde DuckDB"""
    if not DUCKDB_AVAILABLE:
        return None
    
    try:
        conn = duckdb.connect(db_path)

        # Obtener la última fecha disponible en la tabla
        last_date = conn.execute("SELECT MAX(ts_origin) FROM copper_data").fetchone()[0]
        if last_date is None:
            st.warning("La tabla copper_data está vacía")
            return None

        query = f"""
        SELECT 
            ts_origin,
            tag_id,
            value_ts
        FROM copper_data 
        WHERE ts_origin >= DATE('{last_date}') - INTERVAL '{days_back} days'
        AND tag_id IN ('PowerBi.COU1CD2001CU', 'PowerBi.COU1RD001CU', 
                       'PowerBi.COU1CF001CU', 'PowerBi.COU1-RCT-CU')
        ORDER BY ts_origin, tag_id
        """

        df = conn.execute(query).df()
        conn.close()
        
        if len(df) > 0:
            df['ts_origin'] = pd.to_datetime(df['ts_origin'])
            df['ts'] = df['ts_origin']
        
        return df
    
    except Exception as e:
        st.error(f"Error conectando a DuckDB: {e}")
        return None


def create_15min_blocks(df):
    """Crea bloques de 15 minutos"""
    df['time_block'] = df['ts_origin'].dt.floor('15min')
    df_blocks = df.groupby(['time_block', 'tag_id'])['value_ts'].mean().reset_index()
    df_pivot = df_blocks.pivot(index='time_block', columns='tag_id', values='value_ts').reset_index()
    
    column_mapping = {
        'PowerBi.COU1CD2001CU': 'concentrado_cu',
        'PowerBi.COU1RD001CU': 'colas_cu', 
        'PowerBi.COU1CF001CU': 'alimentacion_cu',
        'PowerBi.COU1-RCT-CU': 'recuperacion_sistema'
    }
    
    df_pivot.rename(columns=column_mapping, inplace=True)
    return df_pivot.dropna()

def calculate_manual_recovery(df):
    """Calcula recuperación manual sin filtros extremos"""
    colas = df['colas_cu'] / 10000
    
    # Aplicar fórmula
    numerador = (df['concentrado_cu'] - colas) * df['alimentacion_cu']
    denominador = (df['alimentacion_cu'] - colas) * df['concentrado_cu']
    
    # Solo evitar división exacta por cero
    df['recuperacion_manual'] = np.where(denominador != 0, (numerador / denominador) * 100, np.nan)
    df['diferencia'] = df['recuperacion_manual'] - df['recuperacion_sistema']
    df['diferencia_abs'] = abs(df['diferencia'])
    
    return df

def create_stats(df_filtered, threshold):
    """Calcula estadísticas"""
    # Reemplazar infinitos con NaN
    df_clean = df_filtered.replace([np.inf, -np.inf], np.nan)
    
    # Filtrar solo valores finitos
    valid_diff = df_clean['diferencia_abs'].dropna()
    valid_manual = df_clean['recuperacion_manual'].dropna()
    valid_sistema = df_clean['recuperacion_sistema'].dropna()
    
    if len(valid_diff) == 0:
        return {
            'avg_diff': 0,
            'max_diff': 0,
            'alerts_count': 0,
            'total_points': len(df_filtered),
            'alert_pct': 0,
            'avg_recovery': 0,
            'correlation': 0
        }
    
    alerts = len(valid_diff[valid_diff > threshold])
    
    # Calcular correlación solo con datos válidos en ambas columnas
    common_idx = df_clean[['recuperacion_manual', 'recuperacion_sistema']].dropna().index
    corr_data = df_clean.loc[common_idx]
    correlation = corr_data['recuperacion_manual'].corr(corr_data['recuperacion_sistema']) if len(corr_data) > 1 else 0
    
    return {
        'avg_diff': valid_diff.mean(),
        'max_diff': valid_diff.max(),
        'alerts_count': alerts,
        'total_points': len(valid_diff),
        'alert_pct': (alerts/len(valid_diff)*100) if len(valid_diff) > 0 else 0,
        'avg_recovery': valid_manual.mean(),
        'correlation': correlation if not np.isnan(correlation) else 0
    }

def create_main_chart(df_filtered, threshold):
    """Gráfico principal con zoom"""
    # Limpiar infinitos para el gráfico
    df_plot = df_filtered.replace([np.inf, -np.inf], np.nan)
    
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Recuperación: Manual vs Sistema', 'Diferencia Absoluta'),
                       vertical_spacing=0.12, row_heights=[0.7, 0.3])
    
    # Líneas principales
    fig.add_trace(go.Scatter(x=df_plot['time_block'], y=df_plot['recuperacion_manual'],
                            mode='lines', name='Manual', line=dict(color='#1f77b4', width=1.5),
                            hovertemplate='Manual: %{y:.2f}%<br>%{x}<extra></extra>'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_plot['time_block'], y=df_plot['recuperacion_sistema'],
                            mode='lines', name='Sistema', line=dict(color='#ff7f0e', width=1.5),
                            hovertemplate='Sistema: %{y:.2f}%<br>%{x}<extra></extra>'), row=1, col=1)
    
    # Diferencias con colores
    valid_diff = df_plot['diferencia_abs'].dropna()
    if len(valid_diff) > 0:
        colors = ['red' if x > threshold else 'green' for x in valid_diff]
        fig.add_trace(go.Scatter(x=df_plot.loc[valid_diff.index, 'time_block'], y=valid_diff,
                                mode='markers', name='Diferencia', marker=dict(color=colors, size=3, opacity=0.7),
                                hovertemplate='Diferencia: %{y:.3f}%<br>%{x}<extra></extra>'), row=2, col=1)
    
    # Umbral
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Umbral: {threshold}%", row=2, col=1)
    
    fig.update_layout(title="Análisis Temporal de Recuperación de Cobre", height=700,
                     showlegend=True, hovermode='x unified', dragmode='zoom', selectdirection='h')
    
    fig.update_xaxes(title_text="Tiempo", row=2, col=1, rangeslider=dict(visible=False), type='date', matches='x')
    fig.update_yaxes(title_text="Recuperación (%)", row=1, col=1)
    fig.update_yaxes(title_text="Diferencia Absoluta (%)", row=2, col=1)
    
    return fig

def create_analysis_charts(df_filtered):
    """Gráficos de análisis"""
    df_clean = df_filtered.replace([np.inf, -np.inf], np.nan)
    col1, col2 = st.columns(2)
    
    with col1:
        valid_diff = df_clean['diferencia'].dropna()
        if len(valid_diff) > 0:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=valid_diff, nbinsx=50, name='Distribución',
                                           marker_color='skyblue', opacity=0.7))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.update_layout(title="Distribución de Diferencias", xaxis_title="Diferencia (%)",
                                  yaxis_title="Frecuencia", height=400)
            st.plotly_chart(fig_hist, width='stretch')
    
    with col2:
        # Filtrar datos válidos para scatter
        scatter_data = df_clean[['recuperacion_sistema', 'recuperacion_manual', 'diferencia_abs']].dropna()
        if len(scatter_data) > 0:
            fig_scatter = px.scatter(scatter_data, x='recuperacion_sistema', y='recuperacion_manual',
                                   color='diferencia_abs', size='diferencia_abs', color_continuous_scale='Viridis',
                                   title="Correlación Manual vs Sistema",
                                   labels={'recuperacion_sistema': 'Sistema (%)', 'recuperacion_manual': 'Manual (%)',
                                          'diferencia_abs': 'Diferencia (%)'})
            
            # Línea perfecta
            min_val = min(scatter_data['recuperacion_sistema'].min(), scatter_data['recuperacion_manual'].min())
            max_val = max(scatter_data['recuperacion_sistema'].max(), scatter_data['recuperacion_manual'].max())
            fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                            name='Correlación Perfecta', line=dict(dash='dash', color='red')))
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, width='stretch')

def create_time_analysis(df_filtered):
    """Análisis temporal"""
    df_clean = df_filtered.replace([np.inf, -np.inf], np.nan)
    df_temp = df_clean.dropna(subset=['diferencia_abs']).copy()
    
    if len(df_temp) == 0:
        st.warning("No hay datos válidos para análisis temporal")
        return
        
    df_temp['hour'] = df_temp['time_block'].dt.hour
    df_temp['date'] = df_temp['time_block'].dt.date
    
    col1, col2 = st.columns(2)
    
    with col1:
        hourly_stats = df_temp.groupby('hour')['diferencia_abs'].agg(['mean', 'std']).reset_index()
        if len(hourly_stats) > 0:
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['mean'],
                                         mode='lines+markers', name='Promedio', line=dict(color='blue')))
            fig_hour.update_layout(title="Diferencias por Hora del Día", xaxis_title="Hora",
                                  yaxis_title="Diferencia Promedio (%)", height=400)
            st.plotly_chart(fig_hour, width='stretch')
    
    with col2:
        daily_stats = df_temp.groupby('date')['diferencia_abs'].mean().reset_index()
        if len(daily_stats) > 0:
            fig_daily = px.line(daily_stats, x='date', y='diferencia_abs', title="Tendencia Diaria de Diferencias",
                               labels={'date': 'Fecha', 'diferencia_abs': 'Diferencia Promedio (%)'})
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, width='stretch')

def generate_report(df_filtered, stats, threshold):
    """Genera reporte Excel"""
    output = BytesIO()
    df_clean = df_filtered.replace([np.inf, -np.inf], np.nan)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Datos principales
        df_clean[['time_block', 'recuperacion_manual', 'recuperacion_sistema', 
                  'diferencia', 'diferencia_abs']].to_excel(writer, sheet_name='Datos', index=False)
        
        # Estadísticas
        pd.DataFrame([stats]).T.to_excel(writer, sheet_name='Estadísticas')
        
        # Alertas
        valid_diff = df_clean['diferencia_abs'].dropna()
        alerts = df_clean.loc[valid_diff.index][valid_diff > threshold]
        if len(alerts) > 0:
            alerts[['time_block', 'recuperacion_manual', 'recuperacion_sistema', 'diferencia']].to_excel(
                writer, sheet_name='Alertas', index=False)
    
    return output.getvalue()

def create_dashboard(df):
    """Dashboard principal"""
    st.title("Panel de Control - Recuperación de Cobre")
    st.markdown("### Atalaya Mining - Análisis de Discrepancias en Tiempo Real")
    st.divider()
    
    st.sidebar.header("Controles de Análisis")
    
    if len(df) == 0:
        st.error("No hay datos para mostrar")
        return
    
    # Filtros
    date_range = st.sidebar.date_input("Rango de fechas:",
                                      value=(df['time_block'].min().date(), df['time_block'].max().date()),
                                      min_value=df['time_block'].min().date(),
                                      max_value=df['time_block'].max().date())
    
    threshold = st.sidebar.slider("Umbral de alerta (%):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    # Aplicar filtros
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['time_block'].dt.date >= start_date) & 
                        (df['time_block'].dt.date <= end_date)].copy()
    else:
        df_filtered = df.copy()
    
    # KPIs
    stats = create_stats(df_filtered, threshold)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1: st.metric("Diferencia Promedio", f"{stats['avg_diff']:.3f}%")
    with col2: st.metric("Diferencia Máxima", f"{stats['max_diff']:.3f}%")
    with col3: st.metric("Alertas", f"{stats['alerts_count']}/{stats['total_points']}")
    with col4: st.metric("Recuperación Promedio", f"{stats['avg_recovery']:.2f}%")
    with col5: st.metric("Correlación", f"{stats['correlation']:.3f}")
    
    st.divider()
    
    # Gráficos
    st.plotly_chart(create_main_chart(df_filtered, threshold), width='stretch')
    st.info("Usar zoom: Arrastra para hacer zoom en el gráfico. Doble clic para resetear la vista.")
    
    st.subheader("Análisis Detallado")
    create_analysis_charts(df_filtered)
    
    st.subheader("Patrones Temporales")
    create_time_analysis(df_filtered)
    
    # Alertas
    st.subheader("Alertas Recientes")
    df_clean = df_filtered.replace([np.inf, -np.inf], np.nan)
    valid_alerts = df_clean['diferencia_abs'].dropna()
    alerts_df = df_clean.loc[valid_alerts.index][valid_alerts > threshold]
    
    if len(alerts_df) > 0:
        alerts_display = alerts_df[['time_block', 'recuperacion_manual', 'recuperacion_sistema', 'diferencia']].copy()
        alerts_display['time_block'] = alerts_display['time_block'].dt.strftime('%Y-%m-%d %H:%M')
        alerts_display.columns = ['Fecha/Hora', 'Manual (%)', 'Sistema (%)', 'Diferencia (%)']
        st.dataframe(alerts_display.round(4), width='stretch', hide_index=True)
    else:
        st.success("No hay alertas en el período seleccionado")
    
    # Exportación
    st.subheader("Exportar Reporte")
    if st.button("Generar Reporte Excel"):
        excel_data = generate_report(df_filtered, stats, threshold)
        st.download_button(
            label="Descargar Reporte",
            data=excel_data,
            file_name=f"reporte_recuperacion_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

@st.cache_data
def load_data_cached(_file_content, filename):
    """Carga datos con cache"""
    with open(f"temp_{filename}", "wb") as f:
        f.write(_file_content)
    
    df_raw = load_and_process_data(f"temp_{filename}")
    if df_raw is not None:
        df_blocks = create_15min_blocks(df_raw)
        df_final = calculate_manual_recovery(df_blocks)
        return df_final, len(df_raw)
    return None, 0

def main():
    st.sidebar.header("Cargar Datos")

    DATA_DIR = "/app/data"
    
    # Verificar qué opciones están disponibles
    local_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".csv", ".xlsx"))] if os.path.exists(DATA_DIR) else []
    duckdb_exists = DUCKDB_AVAILABLE and os.path.exists(os.path.join(DATA_DIR, "scada_recovery.duckdb"))
    
    # Crear opciones
    options = []
    if duckdb_exists:
        options.append("Base de datos DuckDB")
    if local_files:
        options.append("Archivos en servidor")
    options.append("Subir archivo")
    
    option = st.sidebar.radio("Fuente de datos:", options)
    
    df_final, raw_count = None, 0

    if option == "Base de datos DuckDB" and duckdb_exists:
        # Selector de días de histórico para DuckDB
        days_back = st.sidebar.selectbox("Días de histórico:", [7, 15, 30, 60, 90], index=2)
        
        # Botón para refrescar datos
        if st.sidebar.button("Actualizar datos"):
            st.cache_data.clear()
        
        with st.spinner("Cargando datos desde DuckDB..."):
            df_raw = load_from_duckdb(days_back)
            if df_raw is not None and len(df_raw) > 0:
                df_blocks = create_15min_blocks(df_raw)
                df_final = calculate_manual_recovery(df_blocks)
                raw_count = len(df_raw)
            else:
                st.error("No se encontraron datos en DuckDB")

    elif option == "Archivos en servidor" and local_files:
        selected_file = st.sidebar.selectbox("Selecciona archivo:", local_files)
        if selected_file:
            file_path = os.path.join(DATA_DIR, selected_file)
            with st.spinner("Procesando datos..."):
                df_raw = load_and_process_data(file_path)
                if df_raw is not None:
                    df_blocks = create_15min_blocks(df_raw)
                    df_final = calculate_manual_recovery(df_blocks)
                    raw_count = len(df_raw)

    elif option == "Subir archivo":
        uploaded_file = st.sidebar.file_uploader("Archivo de datos SCADA:", type=['csv', 'xlsx', 'xls'],
                                                help="Formatos soportados: CSV, Excel")
        if uploaded_file is not None:
            with st.spinner("Procesando datos..."):
                df_final, raw_count = load_data_cached(uploaded_file.getbuffer(), uploaded_file.name)

    # Mostrar dashboard si hay datos
    if df_final is not None:
        st.success(f"Datos cargados: {len(df_final)} bloques de 15 min procesados")

        st.sidebar.write("**Dataset Info:**")
        st.sidebar.write(f"• Registros originales: {raw_count:,}")
        st.sidebar.write(f"• Bloques 15min: {len(df_final):,}")
        st.sidebar.write(f"• Período: {df_final['time_block'].min().strftime('%d/%m/%Y')} - {df_final['time_block'].max().strftime('%d/%m/%Y')}")
        
        # Mostrar fuente de datos
        if option == "Base de datos DuckDB":
            st.sidebar.success("Fuente: DuckDB")
        elif option == "Archivos en servidor":
            st.sidebar.info("Fuente: Archivo local")
        else:
            st.sidebar.info("Fuente: Archivo subido")

        create_dashboard(df_final)
    else:
        st.info("Selecciona una fuente de datos para comenzar el análisis")

        with st.expander("Información del Sistema"):
            st.markdown("""
            **Función:** Análisis de discrepancias entre recuperación calculada vs sistema SCADA

            **Fórmula:** [[(COU1CD2001 - COU1RD001) * COU1CF001] / [ (COU1CF001 - COU1RD001) * COU1CD2001]] *100

            **Tags requeridos:**
            - COU1CD2001CU 
            - COU1RD001CU (Se divide por 10,000)
            - COU1CF001CU 
            - COU1-RCT-CU (Recuperación sistema)

            **Fuentes disponibles:**
            """)
            
            if duckdb_exists:
                st.write("- Base de datos DuckDB")
            if local_files:
                st.write("- Archivos en servidor")
            st.write("- Subir archivo manual")
            
            st.write("**Impacto:** 0.1% diferencia = miles de euros en pérdidas")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
from pathlib import Path

# ==============================
# 📌 CONFIGURACIÓN INICIAL
# ==============================

st.set_page_config(page_title="Jornada Laboral Conductores", layout="wide")

# ==============================
# 🌍 GEO OFFLINE - CARGAR MUNICIPIOS
# ==============================

@st.cache_data
def cargar_municipios():
    """Carga y valida el archivo de municipios de Colombia"""
    archivo_municipios = Path("municipios_colombia.csv")
    
    if not archivo_municipios.exists():
        st.error(f"❌ No se encuentra el archivo {archivo_municipios}")
        st.stop()
    
    df_mun = pd.read_csv(archivo_municipios)
    
    # Validar columnas necesarias
    columnas_requeridas = ["Latitud", "Longitud", "Municipio", "Departamento"]
    for col in columnas_requeridas:
        if col not in df_mun.columns:
            st.error(f"❌ El archivo de municipios debe tener la columna '{col}'")
            st.stop()
    
    df_mun["Latitud"] = pd.to_numeric(df_mun["Latitud"], errors="coerce")
    df_mun["Longitud"] = pd.to_numeric(df_mun["Longitud"], errors="coerce")
    df_mun = df_mun.dropna(subset=["Latitud", "Longitud"])
    
    if df_mun.empty:
        st.error("❌ No hay datos válidos de municipios")
        st.stop()
    
    return df_mun

# Cargar municipios
municipios_df = cargar_municipios()

# ==============================
# 🎨 INTERFAZ DE USUARIO
# ==============================

st.title("🚛 Jornada Laboral Conductores")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    HORAS_MAX_JORNADA = st.number_input("⏰ Horas máximas jornada", value=8.0, min_value=1.0, max_value=24.0, step=0.5)
    MIN_PAUSA = st.number_input("☕ Pausa mínima (minutos)", value=34, min_value=1, max_value=120, step=5)
    HORAS_MIN_PAUSA = MIN_PAUSA / 60

with col2:
    HORAS_DESCANSO_LARGO = st.number_input("🌙 Horas descanso largo", value=4.0, min_value=1.0, max_value=12.0, step=0.5)
    MIN_PARADA = st.number_input("🚏 Duración mínima parada (minutos)", value=17, min_value=1, max_value=60, step=5)
    UMBRAL_PARADA_MIN = MIN_PARADA / 60

st.markdown("---")

# ==============================
# 📂 LECTOR INTELIGENTE DE ARCHIVOS
# ==============================

def detectar_columna_coordenadas(df):
    """Detecta automáticamente la columna que contiene coordenadas"""
    # Posibles nombres de columna para coordenadas
    posibles_nombres = [
        'Localización', 'Localizacion', 'localización', 'localizacion',
        'Coordenadas', 'coordenadas', 'Ubicación', 'ubicación', 
        'Ubicacion', 'ubicacion', 'GPS', 'Gps', 'gps',
        'Latitud/Longitud', 'lat_lon', 'LatLon'
    ]
    
    # Buscar coincidencias exactas o parciales
    for col in df.columns:
        col_limpio = col.strip()
        if col_limpio in posibles_nombres:
            return col_limpio
        # Buscar coincidencia parcial (ignorando mayúsculas)
        for nombre in posibles_nombres:
            if nombre.lower() in col_limpio.lower():
                return col_limpio
    
    # Si no encuentra, buscar cualquier columna que pueda contener coordenadas
    for col in df.columns:
        # Revisar primeras filas no nulas
        muestra = df[col].dropna()
        if len(muestra) > 0:
            primer_valor = str(muestra.iloc[0])
            # Si tiene formato de coordenadas (número, número)
            if ',' in primer_valor and any(c.isdigit() for c in primer_valor):
                if st.sidebar.checkbox(f"¿Usar '{col}' como coordenadas?", value=True):
                    return col
    
    return None

@st.cache_data
def leer_archivo(file):
    """Lee archivos CSV o Excel con detección automática de formato"""
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            # Intentar diferentes separadores y codificaciones
            for sep, encoding in [(";", "utf-8"), (",", "utf-8"), (";", "latin1"), (",", "latin1")]:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=sep, encoding=encoding)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            else:
                return None
        
        # Limpiar nombres de columnas
        df.columns = df.columns.astype(str).str.strip().str.replace('"', '')
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
        
        # DEBUG: Mostrar columnas encontradas
        st.sidebar.write(f"📁 {file.name} - Columnas: {list(df.columns)}")
        
        # Detectar columna de coordenadas
        col_coordenadas = detectar_columna_coordenadas(df)
        
        if col_coordenadas:
            # Renombrar a 'Localización' estándar
            df = df.rename(columns={col_coordenadas: "Localización"})
            st.sidebar.success(f"✅ {file.name}: Usando columna '{col_coordenadas}' como coordenadas")
        else:
            st.sidebar.warning(f"⚠️ {file.name}: No se encontró columna de coordenadas")
            # Crear columna vacía
            df["Localización"] = ""
        
        return df if not df.empty else None
    
    except Exception as e:
        st.sidebar.error(f"Error en {file.name}: {e}")
        return None

# ==============================
# 📍 FUNCIONES GEOGRÁFICAS
# ==============================

def parse_coords(coord):
    """Parsea coordenadas del formato 'lat, lon' con múltiples variaciones"""
    if pd.isna(coord) or coord == "" or coord is None:
        return np.nan, np.nan
    
    try:
        # Limpiar la cadena
        coord_str = str(coord).strip().replace('"', '').replace("'", "")
        
        # Reemplazar separadores comunes
        coord_str = coord_str.replace(';', ',').replace('|', ',')
        
        # Manejar formato con grados (ej: "4.60971, -74.08175")
        # Buscar patrón de números con decimales separados por coma
        patron = r'(-?\d+\.?\d*)\s*[,;]\s*(-?\d+\.?\d*)'
        match = re.search(patron, coord_str)
        
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            
            # Validar rangos aproximados (Colombia está entre -4 a 13 lat, -80 a -66 lon)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        
        # Intentar split simple
        if ',' in coord_str:
            partes = coord_str.split(',')
            if len(partes) >= 2:
                lat = float(partes[0].strip())
                lon = float(partes[1].strip())
                return lat, lon
                
    except Exception as e:
        print(f"Error parseando '{coord}': {e}")
    
    return np.nan, np.nan

def distancia_metros(lat1, lon1, lat2, lon2):
    """Calcula distancia en metros usando fórmula de Haversine"""
    if np.isnan(lat1) or np.isnan(lat2):
        return float('inf')
    
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def coord_a_municipio(lat, lon):
    """Convierte coordenadas a nombre de municipio y departamento"""
    if np.isnan(lat) or np.isnan(lon):
        return ""
    
    # Cálculo vectorizado del municipio más cercano
    dist = (
        (municipios_df["Latitud"] - lat)**2 +
        (municipios_df["Longitud"] - lon)**2
    )
    
    idx = dist.idxmin()
    row = municipios_df.loc[idx]
    
    return f"{row['Municipio']}, {row['Departamento']}"

# ==============================
# 🔥 CLUSTERING MEJORADO
# ==============================

def clusterizar_ubicaciones(df, radio=300):
    """Agrupa ubicaciones cercanas usando clustering por proximidad"""
    if df.empty:
        return []
    
    clusters = []
    
    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        if np.isnan(lat):
            continue
        
        asignado = False
        
        for c in clusters:
            d = distancia_metros(lat, lon, c["lat"], c["lon"])
            
            if d < radio:
                total_peso = c["peso"] + row["peso"]
                if total_peso > 0:
                    c["lat"] = (c["lat"] * c["peso"] + lat * row["peso"]) / total_peso
                    c["lon"] = (c["lon"] * c["peso"] + lon * row["peso"]) / total_peso
                c["peso"] = total_peso
                c["count"] += 1
                asignado = True
                break
        
        if not asignado:
            clusters.append({
                "lat": lat,
                "lon": lon,
                "peso": row["peso"],
                "count": 1
            })
    
    return clusters

def obtener_ubic_principal(grupo):
    """Determina la ubicación principal de un grupo de registros"""
    if grupo.empty:
        return ""
    
    # Verificar si existe columna Localización
    if "Localización" not in grupo.columns:
        return ""
    
    # Parsear coordenadas
    coords_parseadas = grupo["Localización"].apply(parse_coords)
    grupo_copy = grupo.copy()
    grupo_copy["lat"] = coords_parseadas.str[0]
    grupo_copy["lon"] = coords_parseadas.str[1]
    
    # Calcular pesos
    grupo_copy["peso"] = np.where(
        grupo_copy["estado"].isin(["ralenti", "apagado"]),
        grupo_copy["delta_horas"] * 2,
        grupo_copy["delta_horas"] * 0.3
    )
    
    # Filtrar coordenadas válidas
    grupo_valid = grupo_copy.dropna(subset=["lat", "lon"])
    
    if grupo_valid.empty:
        return ""
    
    # DEBUG: Mostrar cuántas coordenadas válidas hay
    # st.sidebar.write(f"Coordenadas válidas: {len(grupo_valid)} de {len(grupo)}")
    
    # Clustering
    clusters = clusterizar_ubicaciones(grupo_valid)
    
    if not clusters:
        return ""
    
    # Seleccionar el cluster con mayor peso
    mejor = max(clusters, key=lambda x: x["peso"])
    
    return coord_a_municipio(mejor["lat"], mejor["lon"])

# ==============================
# 📤 SUBIR ARCHIVOS
# ==============================

files = st.file_uploader(
    "📁 Sube archivos de seguimiento (CSV o Excel)",
    accept_multiple_files=True,
    type=['csv', 'xlsx']
)

if not files:
    st.info("👆 Sube uno o más archivos para comenzar el análisis")
    st.stop()

# ==============================
# 🔄 PROCESAMIENTO DE DATOS
# ==============================

with st.spinner("🔄 Procesando archivos..."):
    lista_df = []
    
    for file in files:
        df_temp = leer_archivo(file)
        
        if df_temp is None or df_temp.empty:
            st.warning(f"⚠️ No se pudo leer el archivo: {file.name}")
            continue
        
        # Renombrar columnas necesarias
        mapeo_columnas = {
            "Fecha y Hora": "fecha_hora",
            "Velocidad": "velocidad",
            "Ignicion*": "ignicion",
            "Conductor": "conductor"
        }
        
        for old, new in mapeo_columnas.items():
            if old in df_temp.columns:
                df_temp = df_temp.rename(columns={old: new})
        
        # Verificar columnas esenciales
        columnas_necesarias = ["fecha_hora", "velocidad", "ignicion"]
        for col in columnas_necesarias:
            if col not in df_temp.columns:
                st.error(f"❌ El archivo {file.name} no tiene la columna '{col}'")
                st.stop()
        
        df_temp["vehiculo"] = file.name[:6].upper()
        lista_df.append(df_temp)
    
    if len(lista_df) == 0:
        st.error("❌ No hay datos válidos para procesar")
        st.stop()
    
    df = pd.concat(lista_df, ignore_index=True)

# ==============================
# 🧹 LIMPIEZA DE DATOS
# ==============================

with st.spinner("🧹 Limpiando datos..."):
    # Convertir fecha/hora
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
    df = df.dropna(subset=["fecha_hora"])
    
    if df.empty:
        st.error("❌ No hay fechas/horas válidas en los datos")
        st.stop()
    
    # Procesar ignición
    df["ignicion_on"] = df["ignicion"].astype(str).str.lower().isin(["encendido", "1", "true", "on"])
    
    # Limpiar velocidad
    df["velocidad"] = (df["velocidad"].astype(str)
                       .str.replace(",", ".", regex=False)
                       .str.extract(r"(\d+\.?\d*)")[0])
    df["velocidad"] = pd.to_numeric(df["velocidad"], errors="coerce").fillna(0)
    
    # Ordenar
    df = df.sort_values(["vehiculo", "fecha_hora"]).reset_index(drop=True)
    
    # Extraer fecha
    df["fecha"] = df["fecha_hora"].dt.date

# ==============================
# 🚦 ASIGNACIÓN DE ESTADOS
# ==============================

with st.spinner("📊 Analizando estados de operación..."):
    df["estado"] = np.select(
        [
            df["ignicion_on"] & (df["velocidad"] > 0),
            df["ignicion_on"] & (df["velocidad"] == 0),
            ~df["ignicion_on"]
        ],
        ["conduciendo", "ralenti", "apagado"],
        default="desconocido"
    )
    
    # Calcular diferencias de tiempo
    df["fecha_siguiente"] = df.groupby("vehiculo")["fecha_hora"].shift(-1)
    df["delta_horas"] = (df["fecha_siguiente"] - df["fecha_hora"]).dt.total_seconds() / 3600
    df["delta_horas"] = df["delta_horas"].fillna(0)
    df["delta_horas"] = df["delta_horas"].clip(upper=24)

# ==============================
# 🧱 CREACIÓN DE BLOQUES
# ==============================

with st.spinner("🔨 Construyendo bloques de actividad..."):
    # Identificar cambios de estado
    df["grupo"] = (df["estado"] != df["estado"].shift()).cumsum()
    
    # Agrupar bloques
    bloques = df.groupby(["vehiculo", "grupo"]).agg({
        "estado": "first",
        "fecha_hora": ["min", "max"],
        "delta_horas": "sum",
        "Localización": ["first", "last"]
    })
    
    bloques.columns = ["estado", "inicio", "fin", "duracion_horas", "inicio_ubica", "fin_ubica"]
    bloques = bloques.reset_index()
    bloques["duracion_horas"] = bloques["duracion_horas"].round(2)

# ==============================
# 📈 CÁLCULO DE KPIs
# ==============================

with st.spinner("📈 Calculando indicadores..."):
    kpis_list = []
    
    for (vehiculo, fecha), grupo in df.groupby(["vehiculo", "fecha"]):
        # Obtener conductor
        conductor = grupo["conductor"].mode()
        conductor_nombre = conductor.iloc[0] if not conductor.empty else "Desconocido"
        
        # Horarios de jornada
        datos_ignicion = grupo[grupo["ignicion_on"]]
        if not datos_ignicion.empty:
            inicio_jornada = datos_ignicion["fecha_hora"].min()
            fin_jornada = datos_ignicion["fecha_hora"].max()
        else:
            inicio_jornada = grupo["fecha_hora"].min()
            fin_jornada = grupo["fecha_hora"].max()
        
        # Sumar tiempos
        horas_conduccion = grupo.loc[grupo["estado"] == "conduciendo", "delta_horas"].sum()
        horas_ralenti = grupo.loc[grupo["estado"] == "ralenti", "delta_horas"].sum()
        horas_trabajo = horas_conduccion + horas_ralenti
        
        # Ubicación final (última coordenada válida)
        ubicacion = ""
        if "Localización" in grupo.columns:
            ultima_ubicacion = grupo["Localización"].dropna()
            if not ultima_ubicacion.empty:
                for coord in reversed(ultima_ubicacion):
                    lat, lon = parse_coords(coord)
                    if not np.isnan(lat):
                        ubicacion = coord_a_municipio(lat, lon)
                        break
        
        # Ubicación principal
        ubic_principal = obtener_ubic_principal(grupo)
        
        # Análisis de paradas
        numero_paradas = 0
        horas_descanso = 0
        horas_pausa = 0
        
        bloques_fecha = bloques[
            (bloques["vehiculo"] == vehiculo) &
            (bloques["inicio"].dt.date == fecha)
        ]
        
        for _, b in bloques_fecha.iterrows():
            horas = b["duracion_horas"]
            
            if b["estado"] in ["ralenti", "apagado"] and horas >= UMBRAL_PARADA_MIN:
                numero_paradas += 1
            
            if b["estado"] == "apagado":
                if horas >= HORAS_DESCANSO_LARGO:
                    horas_descanso += horas
                elif horas >= HORAS_MIN_PAUSA:
                    horas_pausa += horas
        
        kpis_list.append({
            "conductor": conductor_nombre,
            "vehiculo": vehiculo,
            "fecha": fecha,
            "origen": "",
            "destino": "",
            "ubicación": ubicacion,
            "inicio_jornada": inicio_jornada,
            "fin_jornada": fin_jornada,
            "numero_paradas": numero_paradas,
            "horas_trabajo": round(horas_trabajo, 2),
            "horas_conduccion": round(horas_conduccion, 2),
            "horas_descanso": round(horas_descanso, 2),
            "horas_pausa": round(horas_pausa, 2),
            "horas_ralenti": round(horas_ralenti, 2),
            "ubic_principal": ubic_principal
        })
    
    kpis = pd.DataFrame(kpis_list)
    
    if kpis.empty:
        st.error("❌ No se pudieron calcular KPIs")
        st.stop()
    
    # Formatear horas
    kpis["inicio_jornada"] = pd.to_datetime(kpis["inicio_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")
    kpis["fin_jornada"] = pd.to_datetime(kpis["fin_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")

# ==============================
# 📊 VISUALIZACIÓN
# ==============================

st.success(f"✅ Procesados {len(files)} archivos - {len(kpis)} jornadas analizadas")

# Mostrar estadísticas de coordenadas
if "ubicación" in kpis.columns:
    coordenadas_validas = kpis[kpis["ubicación"] != ""].shape[0]
    st.info(f"📍 Ubicaciones encontradas: {coordenadas_validas} de {len(kpis)} jornadas ({coordenadas_validas/len(kpis)*100:.1f}%)")

st.markdown("---")

# Mostrar KPIs
st.subheader("📋 Resumen de Jornadas")

column_config = {
    "conductor": "Conductor",
    "vehiculo": "Vehículo",
    "fecha": "Fecha",
    "ubicación": "Ubicación Final",
    "ubic_principal": "Ubicación Principal",
    "horas_trabajo": st.column_config.NumberColumn("Horas Trabajo", format="%.2f h"),
    "horas_conduccion": st.column_config.NumberColumn("Horas Conducción", format="%.2f h"),
    "horas_ralenti": st.column_config.NumberColumn("Horas Ralentí", format="%.2f h"),
    "horas_descanso": st.column_config.NumberColumn("Horas Descanso", format="%.2f h"),
    "horas_pausa": st.column_config.NumberColumn("Horas Pausa", format="%.2f h"),
    "numero_paradas": "N° Paradas",
}

st.dataframe(kpis, use_container_width=True, column_config=column_config)

# Alertas
jornadas_extensas = kpis[kpis["horas_trabajo"] > HORAS_MAX_JORNADA]
if not jornadas_extensas.empty:
    st.warning(f"⚠️ {len(jornadas_extensas)} jornada(s) superan el límite de {HORAS_MAX_JORNADA} horas")

# Debug info en expander
with st.expander("🔧 Información de depuración"):
    st.write("### Columnas en el archivo original:")
    if 'df' in locals():
        st.write(list(df.columns))
    
    st.write("### Muestra de coordenadas:")
    if 'df' in locals() and "Localización" in df.columns:
        muestra_coords = df["Localización"].dropna().head(10)
        for i, coord in enumerate(muestra_coords):
            lat, lon = parse_coords(coord)
            st.write(f"{i+1}. Original: '{coord}' → Parseado: ({lat}, {lon})")
    else:
        st.warning("No se encontró la columna 'Localización' en los datos")

# ==============================
# 💾 EXPORTAR A EXCEL (igual que antes)
# ==============================

# ... (código de exportación igual al anterior)

st.markdown("---")
st.info("💡 **Nota:** Si las ubicaciones siguen vacías, revisa la sección 'Información de depuración' para ver cómo se están parseando las coordenadas")

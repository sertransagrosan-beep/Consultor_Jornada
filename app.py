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

# Cargar municipios (esto se ejecuta una sola vez)
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
                    if len(df.columns) > 1:  # Si tiene más de una columna, probablemente es correcto
                        break
                except:
                    continue
            else:
                return None
        
        # Limpiar nombres de columnas
        df.columns = df.columns.astype(str).str.strip().str.replace('"', '')
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
        
        return df if not df.empty else None
    
    except Exception as e:
        print(f"Error leyendo archivo {file.name}: {e}")
        return None

# ==============================
# 📍 FUNCIONES GEOGRÁFICAS
# ==============================

def parse_coords(coord):
    """Parsea coordenadas del formato 'lat, lon'"""
    if pd.isna(coord) or coord == "":
        return np.nan, np.nan
    
    try:
        # Limpiar la cadena
        coord_str = str(coord).strip().replace('"', '')
        # Reemplazar coma por punto si es necesario
        coord_str = coord_str.replace(',', '.').replace(';', ',')
        
        partes = coord_str.split(',')
        if len(partes) >= 2:
            lat = float(partes[0].strip())
            lon = float(partes[1].strip())
            return lat, lon
    except:
        pass
    
    return np.nan, np.nan

def distancia_metros(lat1, lon1, lat2, lon2):
    """Calcula distancia en metros usando fórmula de Haversine"""
    if np.isnan(lat1) or np.isnan(lat2):
        return float('inf')
    
    R = 6371000  # Radio terrestre en metros
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
    """
    Agrupa ubicaciones cercanas usando clustering por proximidad
    radio: distancia máxima en metros para considerar el mismo cluster
    """
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
                # Media ponderada
                c["lat"] = (c["lat"] * c["peso"] + lat * row["peso"]) / total_peso if total_peso > 0 else 0
                c["lon"] = (c["lon"] * c["peso"] + lon * row["peso"]) / total_peso if total_peso > 0 else 0
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
    """
    Determina la ubicación principal de un grupo de registros
    usando clustering ponderado por tiempo
    """
    if grupo.empty:
        return ""
    
    # Parsear coordenadas de forma vectorizada
    coords_parseadas = grupo["Coordenadas"].apply(parse_coords)
    grupo["lat"] = coords_parseadas.str[0]
    grupo["lon"] = coords_parseadas.str[1]
    
    # Calcular pesos: más peso para ralentí/apagado (descanso)
    grupo["peso"] = np.where(
        grupo["estado"].isin(["ralenti", "apagado"]),
        grupo["delta_horas"] * 2,  # Mayor peso para paradas
        grupo["delta_horas"] * 0.3  # Menor peso para conducción
    )
    
    # Filtrar coordenadas válidas
    grupo_valid = grupo.dropna(subset=["lat", "lon"])
    
    if grupo_valid.empty:
        return ""
    
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
        
        # Renombrar columnas (flexible para diferentes formatos)
        mapeo_columnas = {
            "Fecha y Hora": "fecha_hora",
            "Velocidad": "velocidad",
            "Ignicion*": "ignicion",
            "Conductor": "conductor",
            "Localización": "Localización"
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
    
    # Limpiar velocidad (manejar formato europeo con coma)
    df["velocidad"] = (df["velocidad"].astype(str)
                       .str.replace(",", ".", regex=False)
                       .str.extract(r"(\d+\.?\d*)")[0])
    df["velocidad"] = pd.to_numeric(df["velocidad"], errors="coerce").fillna(0)
    
    # Ordenar por vehículo y tiempo
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
    df["delta_horas"] = df["delta_horas"].clip(upper=24)  # Limitar a 24 horas máximo

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
    bloques = bloques.reset_index(drop=True)
    bloques["duracion_horas"] = bloques["duracion_horas"].round(2)
    
    # Agregar vehículo y fecha
    bloques["vehiculo"] = bloques["inicio"].dt.strftime("%H:%M").apply(lambda x: "temp")  # Placeholder
    # Recalcular vehículo correctamente
    vehiculo_por_grupo = df.groupby("grupo")["vehiculo"].first()
    bloques["vehiculo"] = bloques.index.map(vehiculo_por_grupo)

# ==============================
# 📈 CÁLCULO DE KPIs
# ==============================

with st.spinner("📈 Calculando indicadores..."):
    kpis_list = []
    
    for (vehiculo, fecha), grupo in df.groupby(["vehiculo", "fecha"]):
        # Obtener conductor (el que más aparece)
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
        
        # Sumar tiempos por estado
        horas_conduccion = grupo.loc[grupo["estado"] == "conduciendo", "delta_horas"].sum()
        horas_ralenti = grupo.loc[grupo["estado"] == "ralenti", "delta_horas"].sum()
        horas_trabajo = horas_conduccion + horas_ralenti
        
        # Ubicación final
        ultima_ubicacion = grupo["Localización"].dropna()
        if not ultima_ubicacion.empty:
            lat, lon = parse_coords(ultima_ubicacion.iloc[-1])
            ubicacion = coord_a_municipio(lat, lon)
        else:
            ubicacion = ""
        
        # Ubicación principal (donde más tiempo pasó)
        ubic_principal = obtener_ubic_principal(grupo)
        
        # Análisis de paradas y descansos
        numero_paradas = 0
        horas_descanso = 0
        horas_pausa = 0
        
        # Filtrar bloques de este vehículo en esta fecha
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
st.markdown("---")

# Mostrar KPIs con formato mejorado
st.subheader("📋 Resumen de Jornadas")

# Configurar columnas para mejor visualización
column_config = {
    "conductor": "Conductor",
    "vehiculo": "Vehículo",
    "fecha": "Fecha",
    "horas_trabajo": st.column_config.NumberColumn("Horas Trabajo", format="%.2f h"),
    "horas_conduccion": st.column_config.NumberColumn("Horas Conducción", format="%.2f h"),
    "horas_ralenti": st.column_config.NumberColumn("Horas Ralentí", format="%.2f h"),
    "horas_descanso": st.column_config.NumberColumn("Horas Descanso", format="%.2f h"),
    "horas_pausa": st.column_config.NumberColumn("Horas Pausa", format="%.2f h"),
    "numero_paradas": "N° Paradas",
}

st.dataframe(kpis, use_container_width=True, column_config=column_config)

# Mostrar alertas si hay jornadas extensas
jornadas_extensas = kpis[kpis["horas_trabajo"] > HORAS_MAX_JORNADA]
if not jornadas_extensas.empty:
    st.warning(f"⚠️ {len(jornadas_extensas)} jornada(s) superan el límite de {HORAS_MAX_JORNADA} horas")

# ==============================
# 💾 EXPORTAR A EXCEL
# ==============================

st.markdown("---")
st.subheader("📥 Exportar Reporte")

def auto_ajustar_columnas(worksheet):
    """Ajusta automáticamente el ancho de las columnas en Excel"""
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        
        for cell in column:
            if cell.value:
                cell_length = len(str(cell.value))
                max_length = max(max_length, cell_length)
        
        adjusted_width = min(max_length + 2, 50)
        worksheet.column_dimensions[column_letter].width = adjusted_width

def limpiar_nombre_archivo(texto):
    """Limpia el texto para usarlo como nombre de archivo"""
    texto = str(texto).strip()
    texto = re.sub(r'[\\/*?:"<>|]', "", texto)
    texto = re.sub(r'\s+', " ", texto)
    return texto[:50]  # Limitar longitud

# Generar nombre del archivo
if not kpis.empty:
    conductores = kpis["conductor"].dropna().unique()
    if len(conductores) == 1:
        nombre_conductor = limpiar_nombre_archivo(conductores[0])
    else:
        nombre_conductor = "MULTIPLE"
    
    vehiculos = kpis["vehiculo"].dropna().unique()
    nombre_vehiculo = limpiar_nombre_archivo(vehiculos[0]) if len(vehiculos) == 1 else ""
    
    fechas = pd.to_datetime(kpis["fecha"])
    fecha_min = fechas.min()
    fecha_max = fechas.max()
    
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    
    if fecha_min == fecha_max:
        fecha_str = f"{meses[fecha_min.month]}"
    else:
        fecha_str = f"{meses[fecha_min.month]}_{fecha_min.day:02d}-{fecha_max.day:02d}"
    
    if nombre_vehiculo:
        nombre_archivo = f"{nombre_conductor}_{nombre_vehiculo}_{fecha_str}.xlsx"
    else:
        nombre_archivo = f"{nombre_conductor}_{fecha_str}.xlsx"
else:
    nombre_archivo = "reporte_jornadas.xlsx"

# Crear archivo Excel
buffer = io.BytesIO()

with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    # Escribir hojas
    kpis.to_excel(writer, sheet_name="Resumen", index=False)
    bloques.to_excel(writer, sheet_name="Bloques", index=False)
    
    # Autoajustar columnas
    auto_ajustar_columnas(writer.sheets["Resumen"])
    auto_ajustar_columnas(writer.sheets["Bloques"])

buffer.seek(0)

# Botón de descarga
st.download_button(
    label="📎 Descargar Reporte Excel",
    data=buffer,
    file_name=nombre_archivo,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

# ==============================
# ℹ️ INFORMACIÓN ADICIONAL
# ==============================

with st.expander("ℹ️ Ayuda y notas"):
    st.markdown("""
    ### 📌 Definiciones:
    - **Conduciendo**: Vehículo encendido y velocidad > 0
    - **Ralentí**: Vehículo encendido pero velocidad = 0
    - **Apagado**: Vehículo apagado
    
    ### 🎯 Criterios utilizados:
    - **Parada mínima**: Duración configurable (por defecto 17 minutos)
    - **Pausa mínima**: Duración configurable (por defecto 34 minutos)
    - **Descanso largo**: Duración configurable (por defecto 4 horas)
    
    ### 📍 Georreferenciación:
    - Se utiliza el archivo `municipios_colombia.csv` para convertir coordenadas a municipios
    - El clustering identifica la ubicación principal donde más tiempo permaneció el vehículo
    """)

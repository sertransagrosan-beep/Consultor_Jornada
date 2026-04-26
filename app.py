import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Jornada Laboral Conductores", layout="wide")
st.title("📊 Jornada Laboral Conductores")

# ==============================
# CONFIGURACIÓN CON SLIDERS
# ==============================

st.sidebar.header("⚙️ Parámetros de Configuración")

HORAS_MAX_JORNADA = st.sidebar.slider(
    "Horas máximas jornada", 
    min_value=4.0, 
    max_value=16.0, 
    value=8.0, 
    step=0.5,
    help="Duración máxima permitida de una jornada laboral"
)

HORAS_DESCANSO_LARGO = st.sidebar.slider(
    "Horas descanso largo", 
    min_value=2.0, 
    max_value=12.0, 
    value=4.0, 
    step=0.5,
    help="Tiempo mínimo para considerar descanso entre jornadas"
)

MIN_PAUSA = st.sidebar.slider(
    "Pausa mínima (minutos)", 
    min_value=5, 
    max_value=60, 
    value=34, 
    step=1,
    help="Duración mínima para contar como pausa dentro de la jornada"
)

MIN_PARADA = st.sidebar.slider(
    "Duración mínima parada (minutos)", 
    min_value=1, 
    max_value=30, 
    value=17, 
    step=1,
    help="Duración mínima para contar como parada durante la conducción"
)

UMBRAL_MIN_CONDUCCION = st.sidebar.slider(
    "Conducción mínima por jornada (minutos)", 
    min_value=1, 
    max_value=30, 
    value=6, 
    step=1,
    help="Tiempo mínimo de conducción para considerar una jornada válida (evita jornadas de 0 horas)"
)

# Convertir a horas
HORAS_MIN_PAUSA = MIN_PAUSA / 60
UMBRAL_PARADA_MIN = MIN_PARADA / 60
UMBRAL_CONDUCCION_HORAS = UMBRAL_MIN_CONDUCCION / 60

st.sidebar.markdown("---")
st.sidebar.info(f"📌 **Resumen de configuración:**\n\n"
                f"• Jornada máxima: {HORAS_MAX_JORNADA} horas\n"
                f"• Descanso largo: {HORAS_DESCANSO_LARGO} horas\n"
                f"• Pausa mínima: {MIN_PAUSA} minutos\n"
                f"• Parada mínima: {MIN_PARADA} minutos\n"
                f"• Conducción mínima: {UMBRAL_MIN_CONDUCCION} minutos")

# ==============================
# 🌍 GEO OFFLINE (CON CSV)
# ==============================

@st.cache_data
def cargar_municipios():
    """Carga el archivo de municipios de Colombia"""
    try:
        df_mun = pd.read_csv("municipios_colombia.csv")
        
        df_mun["Latitud"] = pd.to_numeric(df_mun["Latitud"], errors="coerce")
        df_mun["Longitud"] = pd.to_numeric(df_mun["Longitud"], errors="coerce")
        df_mun = df_mun.dropna(subset=["Latitud", "Longitud"])
        
        return df_mun
    except FileNotFoundError:
        st.error("❌ No se encuentra el archivo 'municipios_colombia.csv'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error cargando municipios: {e}")
        st.stop()

# Cargar municipios
municipios_df = cargar_municipios()
st.success(f"✅ Cargados {len(municipios_df)} municipios colombianos")

def coord_a_municipio(lat, lon):
    """Convierte coordenadas a municipio usando archivo local"""
    
    if pd.isna(lat) or pd.isna(lon):
        return ""

    dist = (
        (municipios_df["Latitud"] - lat)**2 +
        (municipios_df["Longitud"] - lon)**2
    )
    
    idx = dist.idxmin()
    row = municipios_df.loc[idx]
    
    return f"{row['Municipio']}, {row['Departamento']}"

# ==============================
# LECTOR INTELIGENTE
# ==============================

def leer_archivo(file):
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            try:
                df = pd.read_csv(file, sep=";", encoding="utf-8")
            except:
                file.seek(0)
                df = pd.read_csv(file, sep=None, engine="python")

        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

        return df
    except Exception as e:
        st.error(f"Error leyendo archivo {file.name}: {e}")
        return None

# ==============================
# GEO AUX
# ==============================

def parse_coords(coord):
    try:
        if pd.isna(coord) or coord == "":
            return np.nan, np.nan
        coord_str = str(coord).strip()
        coord_str = coord_str.replace(';', ',').replace('|', ',')
        partes = coord_str.split(',')
        if len(partes) >= 2:
            lat = float(partes[0].strip())
            lon = float(partes[1].strip())
            return lat, lon
        return np.nan, np.nan
    except:
        return np.nan, np.nan

def distancia_metros(lat1, lon1, lat2, lon2):
    if np.isnan(lat1) or np.isnan(lat2):
        return float('inf')
    
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

# ==============================
# CLUSTERING MEJORADO
# ==============================

def clusterizar_ubicaciones(df, radio=300):
    """Agrupa ubicaciones cercanas en clusters"""
    clusters = []

    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        if np.isnan(lat) or np.isnan(lon):
            continue

        asignado = False

        for c in clusters:
            d = distancia_metros(lat, lon, c["lat"], c["lon"])

            if d < radio:
                total = c["peso"] + row["peso"]
                
                if total > 0:
                    c["lat"] = (c["lat"] * c["peso"] + lat * row["peso"]) / total
                    c["lon"] = (c["lon"] * c["peso"] + lon * row["peso"]) / total
                    
                c["peso"] = total
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
    """Determina la ubicación principal donde más tiempo pasó el vehículo"""
    
    if grupo.empty:
        return ""
    
    g = grupo.copy()
    
    col_coords = None
    for col in ["Coordenadas", "Localización", "Localizacion", "Ubicación"]:
        if col in g.columns:
            col_coords = col
            break
    
    if col_coords is None:
        return ""
    
    try:
        coords_parseadas = g[col_coords].apply(parse_coords)
        g["lat"] = coords_parseadas.apply(lambda x: x[0])
        g["lon"] = coords_parseadas.apply(lambda x: x[1])
    except Exception:
        return ""

    g["peso"] = g.apply(
        lambda r: max(r["delta_horas"] * 2, 0.1) if r["estado"] in ["ralenti", "apagado"]
        else max(r["delta_horas"] * 0.3, 0.01),
        axis=1
    )

    g = g.dropna(subset=["lat", "lon"])
    
    if len(g) == 0:
        return ""

    if g["peso"].sum() == 0:
        g["peso"] = 1.0 / len(g)

    clusters = clusterizar_ubicaciones(g)

    if len(clusters) == 0:
        return ""

    mejor = max(clusters, key=lambda x: x["peso"])
    
    if np.isnan(mejor["lat"]) or np.isnan(mejor["lon"]):
        return ""

    return coord_a_municipio(mejor["lat"], mejor["lon"])

# ==============================
# SUBIR ARCHIVOS
# ==============================

files = st.file_uploader("📁 Sube archivos (CSV o Excel)", accept_multiple_files=True)

if files:
    lista_df = []

    with st.spinner("📂 Procesando archivos..."):
        for file in files:
            df_temp = leer_archivo(file)

            if df_temp is None or df_temp.empty:
                continue

            df_temp = df_temp.rename(columns={
                "Fecha y Hora": "fecha_hora",
                "Velocidad": "velocidad",
                "Ignicion*": "ignicion",
                "Conductor": "conductor"
            })

            df_temp["vehiculo"] = file.name[:6].upper()
            lista_df.append(df_temp)

    if len(lista_df) == 0:
        st.error("❌ No hay datos válidos")
        st.stop()

    df = pd.concat(lista_df, ignore_index=True)

    # ==============================
    # LIMPIEZA DE DATOS
    # ==============================
    
    with st.spinner("🧹 Limpiando datos..."):
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        df = df.dropna(subset=["fecha_hora"])

        df["ignicion_on"] = df["ignicion"].astype(str).str.lower().isin(["encendido"])

        df["velocidad"] = (
            df["velocidad"].astype(str)
            .str.replace(",", ".", regex=False)
            .str.extract(r"(\d+\.?\d*)")[0]
        )
        df["velocidad"] = pd.to_numeric(df["velocidad"], errors="coerce").fillna(0)

        df = df.sort_values(["vehiculo", "fecha_hora"]).reset_index(drop=True)
        df["fecha"] = df["fecha_hora"].dt.date

        df["estado"] = df.apply(
            lambda r: "conduciendo" if r["ignicion_on"] and r["velocidad"] > 0
            else "ralenti" if r["ignicion_on"]
            else "apagado",
            axis=1
        )

        df["fecha_siguiente"] = df.groupby("vehiculo")["fecha_hora"].shift(-1)
        df["delta_horas"] = (
            df["fecha_siguiente"] - df["fecha_hora"]
        ).dt.total_seconds() / 3600
        df["delta_horas"] = df["delta_horas"].fillna(0)

        df["grupo"] = (df["estado"] != df["estado"].shift()).cumsum()
        
        bloques = df.groupby(["vehiculo", "grupo"]).agg({
            "estado": "first",
            "fecha_hora": ["min", "max"],
            "delta_horas": "sum",
            "Localización": ["first", "last"]
        })
        
        bloques.columns = [
            "estado",
            "inicio",
            "fin",
            "duracion_horas",
            "inicio_ubica",
            "fin_ubica"
        ]
        
        bloques = bloques.reset_index()
        bloques["duracion_horas"] = bloques["duracion_horas"].round(2)

        # ==============================
    # KPIs - MODELO CORRECTO (BLOQUES REALES + 24H)
    # ==============================
    
    kpis_list = []
    
    # 🔥 IMPORTANTE: usar bloques (no registros crudos)
    bloques = bloques.sort_values(["vehiculo", "inicio"]).reset_index(drop=True)
    
    for vehiculo in bloques["vehiculo"].unique():
    
        bloques_v = bloques[bloques["vehiculo"] == vehiculo].copy()
    
        # 🔥 recorrer bloques (NO días)
        for i, b in bloques_v.iterrows():
    
            estado = b["estado"]
            inicio = b["inicio"]
            fin = b["fin"]
            duracion = (fin - inicio).total_seconds() / 3600
    
            if pd.isna(inicio) or pd.isna(fin) or duracion <= 0:
                continue
    
            fecha_inicio = inicio.date()
            fecha_fin = fin.date()
    
            # 🔥 dividir bloque solo si cruza días
            dias = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    
            for dia in dias:
    
                inicio_dia = pd.Timestamp(dia)
                fin_dia = inicio_dia + pd.Timedelta(days=1)
    
                inicio_real = max(inicio, inicio_dia)
                fin_real = min(fin, fin_dia)
    
                if inicio_real >= fin_real:
                    continue
    
                horas = (fin_real - inicio_real).total_seconds() / 3600
    
                key = (vehiculo, dia)
    
                if key not in kpis_list:
                    kpis_list.append({
                        "vehiculo": vehiculo,
                        "fecha": dia,
                        "horas_conduccion": 0,
                        "horas_ralenti": 0,
                        "horas_trabajo": 0,
                        "horas_descanso": 0,
                        "horas_pausa": 0,
                        "numero_paradas": 0
                    })
    
                # encontrar registro existente
                registro = next(x for x in kpis_list if x["vehiculo"] == vehiculo and x["fecha"] == dia)
    
                # ==========================
                # CLASIFICACIÓN CORRECTA
                # ==========================
    
                if estado == "conduciendo":
                    registro["horas_conduccion"] += horas
    
                elif estado == "ralenti":
                    registro["horas_ralenti"] += horas
    
                    if horas >= UMBRAL_PARADA_MIN:
                        registro["numero_paradas"] += 1
    
                elif estado == "apagado":
    
                    if horas >= HORAS_DESCANSO_LARGO:
                        registro["horas_descanso"] += horas
    
                    elif horas >= HORAS_MIN_PAUSA:
                        registro["horas_pausa"] += horas
    
                    # contar parada también en apagado corto
                    if horas >= UMBRAL_PARADA_MIN:
                        registro["numero_paradas"] += 1
    
    # ==============================
    # CONSTRUIR DATAFRAME FINAL
    # ==============================
    
    kpis = pd.DataFrame(kpis_list)
    
    if not kpis.empty:
    
        # 🔥 completar métricas
        kpis["horas_trabajo"] = kpis["horas_conduccion"] + kpis["horas_ralenti"]
    
        # redondeo limpio
        for col in ["horas_conduccion", "horas_ralenti", "horas_trabajo",
                    "horas_descanso", "horas_pausa"]:
            kpis[col] = kpis[col].round(2)
    
        kpis["numero_paradas"] = kpis["numero_paradas"].astype(int)
    
        # ==========================
        # 🔥 INICIO / FIN JORNADA REAL
        # ==========================
    
        jornadas = []
    
        for (vehiculo, fecha), grupo in df.groupby(["vehiculo", "fecha"]):
    
            df_encendido = grupo[grupo["ignicion_on"]]
    
            if df_encendido.empty:
                inicio = ""
                fin = ""
            else:
                inicio = df_encendido["fecha_hora"].min()
                fin = df_encendido["fecha_hora"].max()
    
            jornadas.append({
                "vehiculo": vehiculo,
                "fecha": fecha,
                "inicio_jornada": inicio,
                "fin_jornada": fin,
                "conductor": grupo["conductor"].dropna().iloc[0] if not grupo["conductor"].dropna().empty else "Desconocido",
                "ubic_principal": obtener_ubic_principal(grupo)
            })
    
        jornadas_df = pd.DataFrame(jornadas)
    
        # merge final
        kpis = kpis.merge(jornadas_df, on=["vehiculo", "fecha"], how="left")
    
        # ==========================
        # FORMATO HORAS
        # ==========================
    
        mask = kpis["horas_trabajo"] > 0
    
        kpis.loc[mask, "inicio_jornada"] = pd.to_datetime(
            kpis.loc[mask, "inicio_jornada"]
        ).dt.strftime("%I:%M %p").str.lstrip("0")
    
        kpis.loc[mask, "fin_jornada"] = pd.to_datetime(
            kpis.loc[mask, "fin_jornada"]
        ).dt.strftime("%I:%M %p").str.lstrip("0")
    
        # ordenar
        kpis = kpis.sort_values(["vehiculo", "fecha"]).reset_index(drop=True)

    # ==============================
    # EXPORTAR
    # ==============================
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        kpis.to_excel(writer, sheet_name="Resumen", index=False)
        bloques.to_excel(writer, sheet_name="Bloques", index=False)
    
        ws_resumen = writer.book["Resumen"]
        ws_bloques = writer.book["Bloques"]
    
        def auto_ajustar(ws):
            for col in ws.columns:
                max_length = 0
                col_letter = col[0].column_letter
                for cell in col:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max_length + 2, 40)
    
        auto_ajustar(ws_resumen)
        auto_ajustar(ws_bloques)
    
    buffer.seek(0)
   
    # ==============================
    # NOMBRE ARCHIVO
    # ==============================
    
    def limpiar_texto(txt):
        txt = str(txt).strip()
        txt = " ".join(txt.split())
        txt = re.sub(r'[\\/*?:"<>|]', "", txt)
        return txt
    
    if not kpis.empty:
        conductores = kpis["conductor"].dropna().unique()
        conductor_nombre = limpiar_texto(conductores[0]) if len(conductores) == 1 else "MULTIPLE_CONDUCTOR"
    
        vehiculos = kpis["vehiculo"].dropna().unique()
        vehiculo = limpiar_texto(vehiculos[0]) if len(vehiculos) == 1 else ""
    
        fechas = pd.to_datetime(kpis["fecha"])
        fecha_min = fechas.min()
        fecha_max = fechas.max()
    
        meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                 5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
    
        mes_nombre = meses[fecha_min.month]
    
        if fecha_min == fecha_max:
            fecha_str = mes_nombre
        else:
            fecha_str = f"{mes_nombre} {fecha_min.day:02d}-{fecha_max.day:02d}"
    
        if vehiculo:
            nombre_archivo = f"{conductor_nombre} {vehiculo} {fecha_str}.xlsx"
        else:
            nombre_archivo = f"{conductor_nombre} {fecha_str}.xlsx"
    else:
        nombre_archivo = "reporte.xlsx"
    
    st.download_button(
        "📥 Descargar Excel", 
        data=buffer, 
        file_name=nombre_archivo,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👈 Sube archivos CSV o Excel para comenzar el análisis")

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Jornada Laboral Conductores", layout="wide")
st.title("📊 Jornada Laboral Conductores")

# ==============================
# CONFIGURACIÓN CON SLIDERS
# ==============================

st.sidebar.header("⚙️ Parámetros de Configuración")

HORAS_MAX_JORNADA = st.sidebar.slider(
    "Horas máximas jornada", 
    min_value=0.0,
    max_value=24.0,
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
    min_value=30,
    max_value=120,
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
    value=17,
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
                f"• Descanso largo mínimo: {HORAS_DESCANSO_LARGO} horas\n"
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

    # Cálculo vectorizado del municipio más cercano
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
        # Reemplazar separadores comunes
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
# 🔥 CLUSTERING MEJORADO
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
    
    # Verificar si existe la columna de coordenadas
    col_coords = None
    for col in ["Coordenadas", "Localización", "Localizacion", "Ubicación"]:
        if col in g.columns:
            col_coords = col
            break
    
    if col_coords is None:
        return ""
    
    # Parsear coordenadas
    try:
        coords_parseadas = g[col_coords].apply(parse_coords)
        g["lat"] = coords_parseadas.apply(lambda x: x[0])
        g["lon"] = coords_parseadas.apply(lambda x: x[1])
    except Exception:
        return ""

    # Calcular pesos
    g["peso"] = g.apply(
        lambda r: max(r["delta_horas"] * 2, 0.1) if r["estado"] in ["ralenti", "apagado"]
        else max(r["delta_horas"] * 0.3, 0.01),
        axis=1
    )

    # Filtrar coordenadas válidas
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

            # Renombrar columnas
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

        # ESTADOS
        df["estado"] = df.apply(
            lambda r: "conduciendo" if r["ignicion_on"] and r["velocidad"] > 0
            else "ralenti" if r["ignicion_on"]
            else "apagado",
            axis=1
        )

        # TIEMPOS
        df["fecha_siguiente"] = df.groupby("vehiculo")["fecha_hora"].shift(-1)
        df["delta_horas"] = (
            df["fecha_siguiente"] - df["fecha_hora"]
        ).dt.total_seconds() / 3600
        df["delta_horas"] = df["delta_horas"].fillna(0)

        # BLOQUES (necesario para la formulación original)
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
    # KPIs - FORMULACIÓN ORIGINAL (con sliders conectados)
    # ==============================

    kpis_list = []
    todas_fechas = sorted(df["fecha"].unique())

    with st.spinner("📊 Calculando métricas de jornada..."):
        for vehiculo in df["vehiculo"].unique():

            df_v = df[df["vehiculo"] == vehiculo]
            bloques_v = bloques[bloques["vehiculo"] == vehiculo]

            for fecha in todas_fechas:

                g = df_v[df_v["fecha"] == fecha]
                if g.empty:
                    continue

                df_on = g[g["ignicion_on"]]
                if df_on.empty:
                    continue

                conductor = g["conductor"].dropna().iloc[0] if not g["conductor"].dropna().empty else "Desconocido"

                inicio = df_on["fecha_hora"].min()
                fin = df_on["fecha_hora"].max()
                
                # ==========================================
                # APLICAR FILTRO DE HORAS MÁXIMAS DE JORNADA
                # ==========================================
                duracion_jornada = (fin - inicio).total_seconds() / 3600
                if duracion_jornada > HORAS_MAX_JORNADA:
                    continue  # Saltar jornadas que exceden el máximo permitido

                # ==========================================
                # FÓRMULAS ORIGINALES
                # ==========================================
                
                # horas_conduccion y horas_ralenti
                horas_conduccion = g[g["estado"] == "conduciendo"]["delta_horas"].sum()
                horas_ralenti = g[g["estado"] == "ralenti"]["delta_horas"].sum()
                horas_trabajo = horas_conduccion + horas_ralenti

                # PARADAS / PAUSAS (según lógica original)
                df_mov = df_on.copy().sort_values("fecha_hora")
                df_mov["vel0"] = df_mov["velocidad"] == 0
                df_mov["grp"] = df_mov["vel0"].ne(df_mov["vel0"].shift()).cumsum()

                numero_paradas = 0
                horas_pausa = 0

                for _, gg in df_mov.groupby("grp"):
                    if gg["vel0"].iloc[0]:
                        dur = gg["delta_horas"].sum()
                        if dur >= UMBRAL_PARADA_MIN:
                            numero_paradas += 1
                        if dur >= HORAS_MIN_PAUSA:
                            horas_pausa += dur

                # DESCANSO REAL (usando HORAS_DESCANSO_LARGO como umbral mínimo)
                horas_descanso = 0
                for _, b in bloques_v.iterrows():

                    ini_d = pd.Timestamp(fecha)
                    fin_d = ini_d + pd.Timedelta(days=1)

                    ini_r = max(b["inicio"], ini_d)
                    fin_r = min(b["fin"], fin_d)

                    if ini_r < fin_r and b["estado"] == "apagado":
                        descanso_parcial = (fin_r - ini_r).total_seconds() / 3600
                        # Solo contar si cumple el descanso largo mínimo
                        if descanso_parcial >= HORAS_DESCANSO_LARGO:
                            horas_descanso += descanso_parcial

                # ==========================================
                # UBICACIONES
                # ==========================================
                
                ubicacion = ""
                if "Coordenadas" in g.columns:
                    coords_validas = g["Coordenadas"].dropna()
                    if not coords_validas.empty:
                        lat, lon = parse_coords(coords_validas.iloc[-1])
                        if not np.isnan(lat):
                            ubicacion = coord_a_municipio(lat, lon)

                ubic_principal = obtener_ubic_principal(g)

                kpis_list.append({
                    "conductor": conductor,
                    "vehiculo": vehiculo,
                    "fecha": fecha,
                    "origen": "",
                    "destino": "",
                    "ubicación": ubicacion,
                    "ubic_principal": ubic_principal,
                    "inicio_jornada": inicio,
                    "fin_jornada": fin,
                    "numero_paradas": numero_paradas,
                    "horas_trabajo": round(horas_trabajo, 2),
                    "horas_conduccion": round(horas_conduccion, 2),
                    "horas_descanso": round(horas_descanso, 2),
                    "horas_pausa": round(horas_pausa, 2),
                    "horas_ralenti": round(horas_ralenti, 2)
                })

    # Crear DataFrame de KPIs
    kpis = pd.DataFrame(kpis_list).round(2)

    # Filtrar jornadas con conducción mínima
    if not kpis.empty:
        kpis = kpis[kpis["horas_conduccion"] >= UMBRAL_CONDUCCION_HORAS]

    # Formatear horas
    if not kpis.empty:
        kpis["inicio_jornada"] = pd.to_datetime(kpis["inicio_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")
        kpis["fin_jornada"] = pd.to_datetime(kpis["fin_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")
        
        kpis = kpis.sort_values(["vehiculo", "fecha"])
        
        # Mostrar KPIs
        st.subheader("📋 Resumen de Jornadas")
        st.dataframe(kpis, use_container_width=True)
        
        # Mostrar estadísticas
        st.subheader("📈 Estadísticas")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total jornadas", len(kpis))
        with col2:
            st.metric("Promedio conducción", f"{kpis['horas_conduccion'].mean():.1f} hrs")
        with col3:
            st.metric("Promedio trabajo", f"{kpis['horas_trabajo'].mean():.1f} hrs")
        with col4:
            st.metric("Total paradas", kpis['numero_paradas'].sum())
    
    else:
        st.warning("⚠️ No se encontraron jornadas con conducción efectiva")

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

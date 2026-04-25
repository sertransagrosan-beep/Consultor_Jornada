import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from pathlib import Path

st.title("Jornada Laboral Conductores")

# ==============================
# CONFIGURACIÓN
# ==============================

HORAS_MAX_JORNADA = st.number_input("Horas máximas jornada", value=8.0)
HORAS_DESCANSO_LARGO = st.number_input("Horas descanso largo", value=4.0)

MIN_PAUSA = st.number_input("Pausa mínima (minutos)", value=34)
MIN_PARADA = st.number_input("Duración mínima parada (minutos)", value=17)

HORAS_MIN_PAUSA = MIN_PAUSA / 60
UMBRAL_PARADA_MIN = MIN_PARADA / 60

# ==============================
# 🌍 GEO OFFLINE (CON ARCHIVO CSV)
# ==============================

@st.cache_data
def cargar_municipios():
    """Carga el archivo de municipios de Colombia"""
    archivo = Path("municipios_colombia.csv")
    
    if not archivo.exists():
        st.error(f"❌ No se encuentra el archivo {archivo}")
        st.info("📌 Asegúrate de tener el archivo 'municipios_colombia.csv' en el mismo directorio")
        st.stop()
    
    df_mun = pd.read_csv(archivo)
    
    # Verificar columnas necesarias
    columnas_requeridas = ["Latitud", "Longitud", "Municipio", "Departamento"]
    for col in columnas_requeridas:
        if col not in df_mun.columns:
            st.error(f"❌ El archivo de municipios debe tener la columna '{col}'")
            st.stop()
    
    # Convertir a numérico y limpiar
    df_mun["Latitud"] = pd.to_numeric(df_mun["Latitud"], errors="coerce")
    df_mun["Longitud"] = pd.to_numeric(df_mun["Longitud"], errors="coerce")
    df_mun = df_mun.dropna(subset=["Latitud", "Longitud"])
    
    if df_mun.empty:
        st.error("❌ No hay datos válidos de municipios")
        st.stop()
    
    return df_mun

# Cargar municipios (esto es CRÍTICO - estaba faltando)
municipios_df = cargar_municipios()
st.success(f"✅ Cargados {len(municipios_df)} municipios colombianos")

# ==============================
# 📍 FUNCIÓN DE GEOCODIFICACIÓN OFFLINE
# ==============================

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
    except:
        return None

# ==============================
# GEO AUX
# ==============================

def parse_coords(coord):
    try:
        # Limpiar la cadena de coordenadas
        coord_str = str(coord).strip()
        # Reemplazar coma por punto si es necesario
        coord_str = coord_str.replace(';', ',').replace('|', ',')
        lat, lon = map(float, coord_str.split(","))
        return lat, lon
    except:
        return np.nan, np.nan

def distancia_metros(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ==============================
# 🔥 CLUSTERING MEJORADO
# ==============================

def clusterizar_ubicaciones(df, radio=300):
    clusters = []

    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        if np.isnan(lat):
            continue

        asignado = False

        for c in clusters:
            d = distancia_metros(lat, lon, c["lat"], c["lon"])

            if d < radio:
                total = c["peso"] + row["peso"]

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
    g = grupo.copy()
    
    # VERIFICAR: ¿Existe la columna Coordenadas?
    if "Coordenadas" not in g.columns and "Localización" not in g.columns:
        return ""
    
    # Usar la columna que exista
    col_coords = "Coordenadas" if "Coordenadas" in g.columns else "Localización"
    
    g[["lat", "lon"]] = g[col_coords].apply(lambda x: pd.Series(parse_coords(x)))

    g["peso"] = g.apply(
        lambda r: r["delta_horas"] * 2 if r["estado"] in ["ralenti", "apagado"]
        else r["delta_horas"] * 0.3,
        axis=1
    )

    g = g.dropna(subset=["lat"])

    if len(g) == 0:
        return ""

    clusters = clusterizar_ubicaciones(g)

    if len(clusters) == 0:
        return ""

    mejor = max(clusters, key=lambda x: x["peso"])

    return coord_a_municipio(mejor["lat"], mejor["lon"])

# ==============================
# SUBIR ARCHIVOS
# ==============================

files = st.file_uploader("Sube archivos", accept_multiple_files=True)

if files:
    lista_df = []

    for file in files:
        df_temp = leer_archivo(file)

        if df_temp is None or df_temp.empty:
            continue

        # Renombrar columnas (flexible)
        columnas_originales = df_temp.columns.tolist()
        
        # Buscar columnas por nombre aproximado
        for col in columnas_originales:
            col_lower = col.lower()
            if "fecha" in col_lower and "hora" in col_lower:
                df_temp = df_temp.rename(columns={col: "fecha_hora"})
            elif "velocidad" in col_lower:
                df_temp = df_temp.rename(columns={col: "velocidad"})
            elif "ignicion" in col_lower or "ignición" in col_lower:
                df_temp = df_temp.rename(columns={col: "ignicion"})
            elif "conductor" in col_lower:
                df_temp = df_temp.rename(columns={col: "conductor"})
            elif "coordenada" in col_lower or "localizacion" in col_lower or "ubicacion" in col_lower:
                df_temp = df_temp.rename(columns={col: "Coordenadas"})

        df_temp["vehiculo"] = file.name[:6].upper()
        lista_df.append(df_temp)

    if len(lista_df) == 0:
        st.error("No hay datos válidos")
        st.stop()

    df = pd.concat(lista_df, ignore_index=True)

    # Mostrar columnas encontradas (debug)
    with st.expander("🔍 Columnas encontradas en los archivos"):
        st.write(list(df.columns))

    # LIMPIEZA
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
    df = df.dropna(subset=["fecha_hora"])

    df["ignicion_on"] = df["ignicion"].astype(str).str.lower().isin(["encendido", "1", "true"])

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

    # BLOQUES
    df["grupo"] = (df["estado"] != df["estado"].shift()).cumsum()
    
    # Verificar si existe columna de coordenadas
    col_coords = "Coordenadas" if "Coordenadas" in df.columns else "Localización" if "Localización" in df.columns else None
    
    if col_coords is None:
        st.warning("⚠️ No se encontró una columna de coordenadas. Las ubicaciones aparecerán vacías.")
        # Crear columna dummy
        df["Coordenadas"] = ""
        col_coords = "Coordenadas"
    
    bloques = df.groupby(["vehiculo", "grupo"]).agg({
        "estado": "first",
        "fecha_hora": ["min", "max"],
        "delta_horas": "sum",
        col_coords: ["first", "last"]
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
    # KPIs
    # ==============================

    kpis_list = []

    for (vehiculo, fecha), grupo in df.groupby(["vehiculo", "fecha"]):
        conductor = grupo["conductor"].dropna()
        conductor_nombre = conductor.iloc[0] if not conductor.empty else "Desconocido"

        # Filtrar datos con ignición encendida
        datos_ignicion = grupo[grupo["ignicion_on"]]
        if not datos_ignicion.empty:
            inicio_jornada = datos_ignicion["fecha_hora"].min()
            fin_jornada = datos_ignicion["fecha_hora"].max()
        else:
            inicio_jornada = grupo["fecha_hora"].min()
            fin_jornada = grupo["fecha_hora"].max()

        horas_conduccion = grupo.loc[grupo["estado"] == "conduciendo", "delta_horas"].sum()
        horas_ralenti = grupo.loc[grupo["estado"] == "ralenti", "delta_horas"].sum()
        horas_trabajo = horas_conduccion + horas_ralenti

        # Ubicación final
        ubicacion = ""
        if col_coords in grupo.columns:
            coords_validas = grupo[col_coords].dropna()
            if not coords_validas.empty:
                lat, lon = parse_coords(coords_validas.iloc[-1])
                if not np.isnan(lat):
                    ubicacion = coord_a_municipio(lat, lon)

        # Ubicación principal
        ubic_principal = obtener_ubic_principal(grupo)

        # BLOQUES REALES
        bloques_v = bloques[bloques["vehiculo"] == vehiculo]

        numero_paradas = 0
        horas_descanso = 0
        horas_pausa = 0

        for _, b in bloques_v.iterrows():
            inicio = b["inicio"]
            fin = b["fin"]

            inicio_dia = pd.Timestamp(fecha)
            fin_dia = inicio_dia + pd.Timedelta(days=1)

            inicio_real = max(inicio, inicio_dia)
            fin_real = min(fin, fin_dia)

            if inicio_real < fin_real:
                horas = (fin_real - inicio_real).total_seconds() / 3600

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
            "horas_trabajo": horas_trabajo,
            "horas_conduccion": horas_conduccion,
            "horas_descanso": horas_descanso,
            "horas_pausa": horas_pausa,
            "horas_ralenti": horas_ralenti,
            "ubic_principal": ubic_principal
        })

    kpis = pd.DataFrame(kpis_list).round(2)

    if not kpis.empty:
        kpis["inicio_jornada"] = pd.to_datetime(kpis["inicio_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")
        kpis["fin_jornada"] = pd.to_datetime(kpis["fin_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")

    st.dataframe(kpis)

    # ==============================
    # EXPORTAR
    # ==============================
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        kpis.to_excel(writer, sheet_name="Resumen", index=False)
        bloques.to_excel(writer, sheet_name="Bloques", index=False)
    
        # Autoajuste de columnas
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
        if len(conductores) == 1:
            conductor_nombre = limpiar_texto(conductores[0])
        else:
            conductor_nombre = "MULTIPLE_CONDUCTOR"
    
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
    
    st.download_button("Descargar Excel", data=buffer, file_name=nombre_archivo)

else:
    st.info("📂 Sube archivos para comenzar el análisis")

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Jornada Laboral Conductores", layout="wide")
st.title("📊 Jornada Laboral Conductores")

# ==============================
# CONFIGURACIÓN
# ==============================

st.sidebar.header("⚙️ Parámetros")

HORAS_DESCANSO_LARGO = st.sidebar.slider("Descanso largo (h)", 2.0, 12.0, 4.0)
MIN_PAUSA = st.sidebar.slider("Pausa mínima (min)", 5, 60, 34)
MIN_PARADA = st.sidebar.slider("Parada mínima (min)", 1, 30, 17)

HORAS_MIN_PAUSA = MIN_PAUSA / 60
UMBRAL_PARADA_MIN = MIN_PARADA / 60

# ==============================
# MUNICIPIOS
# ==============================

@st.cache_data
def cargar_municipios():
    df = pd.read_csv("municipios_colombia.csv")
    df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
    return df.dropna()

municipios_df = cargar_municipios()
st.success(f"✅ {len(municipios_df)} municipios cargados")

def coord_a_municipio(lat, lon):
    if pd.isna(lat):
        return ""
    dist = (municipios_df["Latitud"] - lat)**2 + (municipios_df["Longitud"] - lon)**2
    row = municipios_df.loc[dist.idxmin()]
    return f"{row['Municipio']}, {row['Departamento']}"

# ==============================
# HELPERS
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
        return df
    except:
        return None

def parse_coords(coord):
    try:
        lat, lon = map(float, str(coord).replace(";", ",").split(",")[:2])
        return lat, lon
    except:
        return np.nan, np.nan

# ==============================
# CARGA
# ==============================

files = st.file_uploader("📁 Sube archivos", accept_multiple_files=True)

if files:

    dfs = []

    for file in files:
        df = leer_archivo(file)
        if df is None or df.empty:
            continue

        df = df.rename(columns={
            "Fecha y Hora": "fecha_hora",
            "Velocidad": "velocidad",
            "Ignicion*": "ignicion",
            "Conductor": "conductor"
        })

        df["vehiculo"] = file.name[:6].upper()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # ==============================
    # LIMPIEZA
    # ==============================

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
    df = df.dropna(subset=["fecha_hora"])

    df["ignicion_on"] = df["ignicion"].astype(str).str.lower().isin(["encendido"])

    df["velocidad"] = (
        df["velocidad"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
    )
    df["velocidad"] = pd.to_numeric(df["velocidad"], errors="coerce").fillna(0)

    df = df.sort_values(["vehiculo", "fecha_hora"])
    df["fecha"] = df["fecha_hora"].dt.date

    # ==============================
    # ESTADOS
    # ==============================

    df["estado"] = np.where(
        (df["ignicion_on"]) & (df["velocidad"] > 0),
        "conduciendo",
        np.where(df["ignicion_on"], "ralenti", "apagado")
    )

    # ==============================
    # DELTAS
    # ==============================

    df["fecha_sig"] = df.groupby("vehiculo")["fecha_hora"].shift(-1)
    df["delta_horas"] = (
        df["fecha_sig"] - df["fecha_hora"]
    ).dt.total_seconds() / 3600
    df["delta_horas"] = df["delta_horas"].fillna(0)

    # ==============================
    # BLOQUES
    # ==============================

    df["grupo"] = (df["estado"] != df["estado"].shift()).cumsum()

    bloques = df.groupby(["vehiculo", "grupo"]).agg({
        "estado": "first",
        "fecha_hora": ["min", "max"],
        "delta_horas": "sum"
    })

    bloques.columns = ["estado", "inicio", "fin", "duracion_horas"]
    bloques = bloques.reset_index()

    # ==============================
    # KPIs (CORRECTO)
    # ==============================

    kpis_list = []

    for (vehiculo, fecha), grupo in df.groupby(["vehiculo", "fecha"]):

        df_encendido = grupo[grupo["ignicion_on"]]
        if df_encendido.empty:
            continue

        conductor = grupo["conductor"].dropna().iloc[0] if "conductor" in grupo else ""

        inicio_jornada = df_encendido["fecha_hora"].min()
        fin_jornada = df_encendido["fecha_hora"].max()

        horas_conduccion = grupo.loc[grupo["estado"]=="conduciendo","delta_horas"].sum()
        horas_ralenti = grupo.loc[grupo["estado"]=="ralenti","delta_horas"].sum()
        horas_trabajo = horas_conduccion + horas_ralenti

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

                # PARADAS
                if b["estado"] in ["ralenti","apagado"] and horas >= UMBRAL_PARADA_MIN:
                    numero_paradas += 1

                # DESCANSO vs PAUSA (EXCLUYENTE)
                if b["estado"] == "apagado":
                    if horas >= HORAS_DESCANSO_LARGO:
                        horas_descanso += horas
                    elif horas >= HORAS_MIN_PAUSA:
                        horas_pausa += horas

        # UBICACIÓN
        lat, lon = parse_coords(grupo["Coordenadas"].dropna().iloc[-1] if "Coordenadas" in grupo else "")
        ubicacion = coord_a_municipio(lat, lon)

        kpis_list.append({
            "conductor": conductor,
            "vehiculo": vehiculo,
            "fecha": fecha,
            "ubicación": ubicacion,
            "inicio_jornada": inicio_jornada,
            "fin_jornada": fin_jornada,
            "numero_paradas": numero_paradas,
            "horas_trabajo": round(horas_trabajo,2),
            "horas_conduccion": round(horas_conduccion,2),
            "horas_descanso": round(horas_descanso,2),
            "horas_pausa": round(horas_pausa,2),
            "horas_ralenti": round(horas_ralenti,2)
        })

    kpis = pd.DataFrame(kpis_list)

    # ==============================
    # FORMATO FINAL
    # ==============================

    if not kpis.empty:

        kpis["fecha"] = pd.to_datetime(kpis["fecha"], errors="coerce")

        kpis["inicio_jornada"] = pd.to_datetime(kpis["inicio_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")
        kpis["fin_jornada"] = pd.to_datetime(kpis["fin_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")

        st.dataframe(kpis, use_container_width=True)

    # ==============================
    # EXPORTAR
    # ==============================

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        kpis.to_excel(writer, sheet_name="Resumen", index=False)
        bloques.to_excel(writer, sheet_name="Bloques", index=False)

    buffer.seek(0)

    st.download_button(
        "📥 Descargar Excel",
        data=buffer,
        file_name="reporte_jornadas.xlsx"
    )

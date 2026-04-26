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

HORAS_DESCANSO_LARGO = st.sidebar.slider("Descanso largo (horas)", 2.0, 12.0, 4.0, 0.5)
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
st.success(f"✅ Cargados {len(municipios_df)} municipios colombianos")

def coord_a_municipio(lat, lon):
    if pd.isna(lat): return ""
    dist = (municipios_df["Latitud"] - lat)**2 + (municipios_df["Longitud"] - lon)**2
    row = municipios_df.loc[dist.idxmin()]
    return f"{row['Municipio']}, {row['Departamento']}"

# ==============================
# UTILIDADES
# ==============================

def leer_archivo(file):
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            try:
                df = pd.read_csv(file, sep=";")
            except:
                file.seek(0)
                df = pd.read_csv(file, sep=None, engine="python")
        df.columns = df.columns.astype(str).str.strip()
        return df.loc[:, ~df.columns.str.contains("^Unnamed")]
    except:
        return None

def parse_coords(coord):
    try:
        lat, lon = map(float, str(coord).replace(";", ",").split(","))
        return lat, lon
    except:
        return np.nan, np.nan

# ==============================
# UPLOAD
# ==============================

files = st.file_uploader("📁 Sube archivos", accept_multiple_files=True)

if files:

    lista_df = []

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

    df = pd.concat(lista_df, ignore_index=True)

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

    df = df.sort_values(["vehiculo", "fecha_hora"]).reset_index(drop=True)
    df["fecha"] = df["fecha_hora"].dt.date

    df["estado"] = df.apply(
        lambda r: "conduciendo" if r["ignicion_on"] and r["velocidad"] > 0
        else "ralenti" if r["ignicion_on"]
        else "apagado",
        axis=1
    )

    df["fecha_siguiente"] = df.groupby("vehiculo")["fecha_hora"].shift(-1)
    df["delta_horas"] = (df["fecha_siguiente"] - df["fecha_hora"]).dt.total_seconds() / 3600
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
    # KPIs
    # ==============================

    kpis_list = []
    todas_fechas = sorted(df["fecha"].unique())

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

            h_conduccion = g[g["estado"] == "conduciendo"]["delta_horas"].sum()
            h_ralenti = g[g["estado"] == "ralenti"]["delta_horas"].sum()
            h_trabajo = h_conduccion + h_ralenti

            # PARADAS / PAUSAS
            df_mov = df_on.copy().sort_values("fecha_hora")
            df_mov["vel0"] = df_mov["velocidad"] == 0
            df_mov["grp"] = df_mov["vel0"].ne(df_mov["vel0"].shift()).cumsum()

            paradas = 0
            pausas = 0

            for _, gg in df_mov.groupby("grp"):
                if gg["vel0"].iloc[0]:
                    dur = gg["delta_horas"].sum()
                    if dur >= UMBRAL_PARADA_MIN:
                        paradas += 1
                    if dur >= HORAS_MIN_PAUSA:
                        pausas += dur

            # DESCANSO REAL
            descanso = 0
            for _, b in bloques_v.iterrows():

                ini_d = pd.Timestamp(fecha)
                fin_d = ini_d + pd.Timedelta(days=1)

                ini_r = max(b["inicio"], ini_d)
                fin_r = min(b["fin"], fin_d)

                if ini_r < fin_r and b["estado"] == "apagado":
                    descanso += (fin_r - ini_r).total_seconds() / 3600

            ubic = ""
            try:
                lat, lon = parse_coords(g["Coordenadas"].dropna().iloc[-1])
                ubic = coord_a_municipio(lat, lon)
            except:
                pass

            kpis_list.append({
                "conductor": conductor,
                "vehiculo": vehiculo,
                "fecha": fecha,
                "origen": "",
                "destino": "",
                "ubicación": ubic,
                "inicio_jornada": inicio,
                "fin_jornada": fin,
                "numero_paradas": paradas,
                "horas_trabajo": round(h_trabajo, 2),
                "horas_conduccion": round(h_conduccion, 2),
                "horas_descanso": round(descanso, 2),
                "horas_pausa": round(pausas, 2),
                "horas_ralenti": round(h_ralenti, 2),
                "ubic_principal": ubic
            })

    kpis = pd.DataFrame(kpis_list)

    kpis["fecha"] = pd.to_datetime(kpis["fecha"]).dt.date
    kpis = kpis.sort_values(["vehiculo", "fecha"])

    mask = kpis["horas_trabajo"] > 0

    kpis.loc[mask, "inicio_jornada"] = pd.to_datetime(kpis.loc[mask, "inicio_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")
    kpis.loc[mask, "fin_jornada"] = pd.to_datetime(kpis.loc[mask, "fin_jornada"]).dt.strftime("%I:%M %p").str.lstrip("0")

    columnas = [
        "conductor","vehiculo","fecha","origen","destino","ubicación",
        "inicio_jornada","fin_jornada","numero_paradas",
        "horas_trabajo","horas_conduccion","horas_descanso",
        "horas_pausa","horas_ralenti","ubic_principal"
    ]

    kpis = kpis[columnas]

    st.dataframe(kpis, use_container_width=True)

    # ==============================
    # EXPORTAR
    # ==============================

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        kpis.to_excel(writer, sheet_name="Resumen", index=False)
        bloques.to_excel(writer, sheet_name="Bloques", index=False)

    buffer.seek(0)

    st.download_button("📥 Descargar Excel", buffer, "reporte_jornada.xlsx")

else:
    st.info("👈 Sube archivos para comenzar")

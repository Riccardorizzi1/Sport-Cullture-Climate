from pathlib import Path
import json
import tempfile
import zipfile
import gc
import base64
import math
import pyarrow.dataset as pds

import os

import pyproj

PROJ_DATA_DIR = pyproj.datadir.get_data_dir()

os.environ["PROJ_LIB"] = PROJ_DATA_DIR
os.environ["PROJ_DATA"] = PROJ_DATA_DIR

pyproj.datadir.set_data_dir(
    PROJ_DATA_DIR
)

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components

from keplergl import KeplerGl


# ============================================================
# SPORT & CULTURE URBAN CLIMATE
# Dashboard nazionale
# ============================================================

ROOT = Path(__file__).resolve().parent


# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Sport & Culture Urban Climate",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH
# ============================================================

DATA_PROCESSED = (
    ROOT
    / "data"
    / "processed"
)

DATA_OUTPUT = (
    ROOT
    / "data_processed"
    / "output"
)

DATA_URBAN = (
    ROOT
    / "data_processed"
    / "contesto_urbano"
)

DATA_AMBIENTE = (
    ROOT
    / "data_processed"
    / "ambiente"
)

DATA_TECNICO = (
    ROOT
    / "data_processed"
    / "tecnico"
)

LIMITI_ZIP = (
    ROOT
    / "data"
    / "raw"
    / "Limiti01012023.zip"
)


FILES = {

    # STRUTTURE
    "registry":
        DATA_PROCESSED
        / "master_registry_territorio.parquet",

    "registry_500m":
        DATA_PROCESSED
        / "master_registry_griglia_500m_strutture_finale.parquet",

    "osm":
        DATA_PROCESSED
        / "osm_strutture_classificate.parquet",

    "tassonomia":
        DATA_PROCESSED
        / "tassonomia_approvata.json",


    # GRIGLIA 500 M
    "offerta_500m":
        DATA_PROCESSED
        / "griglia_italia_500m_offerta_domanda_finale.parquet",

    "buffer_500m":
        DATA_PROCESSED
        / "griglia_italia_500m_accessibilita_buffer.parquet",

    "contesto_500m":
        DATA_PROCESSED
        / "griglia_italia_500m_contesto_urbano_finale.parquet",

    "ambiente_500m":
        DATA_PROCESSED
        / "griglia_italia_500m_ambiente_finale.parquet",


    # AGGREGAZIONI
    "comuni":
        DATA_PROCESSED
        / "indicatori_comuni_2023.parquet",

    "province":
        DATA_PROCESSED
        / "indicatori_province_2023.parquet",

    "regioni":
        DATA_PROCESSED
        / "indicatori_regioni_2023.parquet",


    # SCORE
    "score_accessibilita":
        DATA_OUTPUT
        / "score_accessibilita_500m.parquet",

    "score_offerta":
        DATA_OUTPUT
        / "score_offerta_domanda_500m.parquet",

    "score_contesto":
        DATA_OUTPUT
        / "score_contesto_urbano_500m.parquet",

    "score_comfort":
        DATA_OUTPUT
        / "score_comfort_500m.parquet",

    "score_attrattivita":
        DATA_OUTPUT
        / "score_attrattivita_territoriale_500m.parquet",

    "indice_finale":
        DATA_OUTPUT
        / "indice_finale_3score_500m.parquet",

    "quartili":
        DATA_OUTPUT
        / "classificazione_quartili_500m.parquet",

    "cluster":
        DATA_OUTPUT
        / "cluster_territoriali_500m.parquet",

    "ranking_regionale":
        DATA_OUTPUT
        / "ranking_regionale.parquet",

    "ranking_provinciale":
        DATA_OUTPUT
        / "ranking_provinciale.parquet",


    # TECNICO
    "registro_tecnico":
        DATA_TECNICO
        / "registro_tecnico_variabili.parquet",
}


# ============================================================
# CONTROLLO FILE ESSENZIALI
# ============================================================

ESSENZIALI = [
    FILES["registry"],
    FILES["contesto_500m"],
    FILES["comuni"],
    FILES["province"],
    FILES["regioni"],
    FILES["score_accessibilita"],
    FILES["registro_tecnico"],
    FILES["tassonomia"],
    LIMITI_ZIP,
]

missing = [
    p
    for p in ESSENZIALI
    if not p.exists()
]

if missing:

    st.error(
        "Mancano alcuni file necessari."
    )

    for p in missing:
        st.code(str(p))

    st.stop()


# ============================================================
# FUNZIONI GENERALI
# ============================================================


def read_parquet(
    path,
    columns=None,
):

    path = Path(path)

    if columns is None:

        return pd.read_parquet(
            path
        )


    dataset = pds.dataset(
        str(path),
        format="parquet",
    )

    available = set(
        dataset.schema.names
    )

    selected = [
        c
        for c in columns
        if c in available
    ]

    if not selected:

        return pd.DataFrame()


    table = dataset.to_table(
        columns=selected
    )

    return table.to_pandas()


def read_parquet_preview(
    path,
    columns,
    n_rows=5000,
):

    """
    Legge solo le prime n_rows necessarie
    alla visualizzazione.

    Non carica l'intero Parquet nazionale.
    """

    dataset = pds.dataset(
        str(path),
        format="parquet",
    )

    available = set(
        dataset.schema.names
    )

    selected = [
        c
        for c in columns
        if c in available
    ]

    if not selected:

        return pd.DataFrame()


    table = dataset.head(
        n_rows,
        columns=selected,
    )

    return table.to_pandas()


@st.cache_data(show_spinner=False)
def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def fmt_int(value):

    if pd.isna(value):
        return "—"

    return (
        f"{int(round(value)):,}"
        .replace(",", ".")
    )


def fmt_float(
    value,
    decimals=2,
):

    if pd.isna(value):
        return "—"

    text = (
        f"{float(value):,.{decimals}f}"
    )

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def normalizza_codice(
    series,
    width,
):

    out = (
        series
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.extract(
            r"(\d+)",
            expand=False,
        )
    )

    return out.str.zfill(
        width
    )


def trova_colonna(
    columns,
    candidates,
):

    mapping = {
        str(c).upper(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.upper() in mapping:

            return mapping[
                candidate.upper()
            ]

    return None


# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# LOGO SBL
# ============================================================

logo_sbl = ROOT / "logo_sbl.png"

if logo_sbl.exists():

    encoded_logo = base64.b64encode(
        logo_sbl.read_bytes()
    ).decode("ascii")

    logo_html = (
        '<a href="https://www.sblconsultancy.it/" '
        'target="_blank" '
        'title="SBL Consultancy" '
        'style="display:block; text-decoration:none; '
        'margin:4px 0 18px 0;">'
        f'<img src="data:image/png;base64,{encoded_logo}" '
        'alt="SBL Consultancy" '
        'style="display:block; width:115px; '
        'max-width:100%; height:auto;">'
        '</a>'
    )

    st.sidebar.markdown(
        logo_html,
        unsafe_allow_html=True,
    )

else:

    st.sidebar.warning(
        "Logo SBL non trovato."
    )


st.sidebar.title(
    "Sport & Culture"
)

st.sidebar.caption(
    "Urban Climate · Italia"
)




st.sidebar.markdown("---")


PAGES = [
    "Panorama nazionale",
    "Offerta sportiva e culturale",
    "Popolazione e territorio",
    "Accessibilità e fruibilità",
    "Contesto urbano e ambiente",
    "Confronti territoriali",
    "Qualità dati e metodologia",
]


pagina = st.sidebar.radio(
    "Sezione",
    PAGES,
    index=0,
)


st.sidebar.markdown("---")


dominio = st.sidebar.radio(
    "Offerta",
    [
        "Entrambi",
        "Sport",
        "Cultura",
    ],
    index=0,
)


st.sidebar.markdown("---")


dark_mode = st.sidebar.toggle(
    "Modalità scura",
    value=False,
)


# ============================================================
# PALETTE
# ============================================================

if dark_mode:

    SIDEBAR_TOP = "#071F31"
    SIDEBAR_MID = "#0E4058"
    SIDEBAR_BOTTOM = "#126982"

    MAIN_LEFT = "#071822"
    MAIN_MID = "#0C2B39"
    MAIN_RIGHT = "#10475B"

    CARD = "#102D3B"

    TEXT = "#F3FAFC"
    TEXT_SECONDARY = "#BBD2DC"

    BORDER = "#315462"

    SHADOW = (
        "rgba(0, 0, 0, 0.28)"
    )

else:

    SIDEBAR_TOP = "#153F57"
    SIDEBAR_MID = "#17627E"
    SIDEBAR_BOTTOM = "#259BBA"

    MAIN_LEFT = "#C7F1F6"
    MAIN_MID = "#9FE5F0"
    MAIN_RIGHT = "#5BC6E2"

    CARD = "#FDFEFF"

    TEXT = "#082B46"
    TEXT_SECONDARY = "#416779"

    BORDER = "#D4E7EC"

    SHADOW = (
        "rgba(8, 43, 70, 0.10)"
    )


# ============================================================
# CSS
#
# SOLO CSS.
# Nessun testo della dashboard viene costruito in HTML.
# ============================================================

st.markdown(
    f"""
<style>

header[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer {{
    display: none !important;
}}


section[data-testid="stSidebar"] {{

    min-width: 350px !important;
    width: 350px !important;
    max-width: 350px !important;

    background:
        linear-gradient(
            180deg,
            {SIDEBAR_TOP} 0%,
            {SIDEBAR_MID} 52%,
            {SIDEBAR_BOTTOM} 100%
        )
        !important;

    border-right:
        1px solid
        rgba(255,255,255,0.15)
        !important;
}}


section[data-testid="stSidebar"] * {{

    color:
        white
        !important;
}}


.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{

    background:
        linear-gradient(
            100deg,
            {MAIN_LEFT} 0%,
            {MAIN_MID} 47%,
            {MAIN_RIGHT} 100%
        )
        !important;

    color:
        {TEXT}
        !important;
}}


.block-container {{

    max-width:
        1600px !important;

    padding-top:
        2.3rem !important;

    padding-bottom:
        4rem !important;

    padding-left:
        3rem !important;

    padding-right:
        3rem !important;
}}


[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] p,
[data-testid="stMain"] label {{

    color:
        {TEXT}
        !important;
}}


[data-testid="stMetric"] {{

    background:
        {CARD}
        !important;

    border:
        1px solid
        {BORDER}
        !important;

    border-radius:
        16px !important;

    padding:
        1.1rem
        1.3rem !important;

    min-height:
        125px !important;

    box-shadow:
        0 5px 14px
        {SHADOW}
        !important;
}}


[data-testid="stMetric"] * {{

    color:
        {TEXT}
        !important;
}}


[data-testid="stDataFrame"] {{

    border:
        1px solid
        {BORDER}
        !important;

    border-radius:
        14px !important;

    overflow:
        hidden !important;
}}


[data-testid="stExpander"] {{

    background:
        {CARD}
        !important;

    border:
        1px solid
        {BORDER}
        !important;

    border-radius:
        14px !important;
}}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# CONTROLLI SIDEBAR
# ============================================================

st.markdown(
    """
<style>

/* Rettangolo del selectbox */
section[data-testid="stSidebar"]
div[data-baseweb="select"] > div {

    background-color: #FFFFFF !important;
    border: 1px solid #D4E7EC !important;
    border-radius: 7px !important;
}


/* Testo selezionato */
section[data-testid="stSidebar"]
div[data-baseweb="select"] span {

    color: #082B46 !important;
    -webkit-text-fill-color: #082B46 !important;
}


/* Eventuale input interno */
section[data-testid="stSidebar"]
div[data-baseweb="select"] input {

    color: #082B46 !important;
    -webkit-text-fill-color: #082B46 !important;
    caret-color: #082B46 !important;
}


/* Freccia */
section[data-testid="stSidebar"]
div[data-baseweb="select"] svg {

    color: #082B46 !important;
    fill: #082B46 !important;
}


/* Menu aperto */
div[data-baseweb="popover"]
[role="listbox"] {

    background-color: #FFFFFF !important;
}


div[data-baseweb="popover"]
[role="option"] {

    background-color: #FFFFFF !important;
    color: #082B46 !important;
}


div[data-baseweb="popover"]
[role="option"] * {

    color: #082B46 !important;
    -webkit-text-fill-color: #082B46 !important;
}


div[data-baseweb="popover"]
[role="option"]:hover {

    background-color: #DDF4F7 !important;
}

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATI AMMINISTRATIVI
# ============================================================

regioni = read_parquet(
    FILES["regioni"]
)

province = read_parquet(
    FILES["province"]
)

comuni = read_parquet(
    FILES["comuni"]
)


# ============================================================
# FILTRI TERRITORIALI
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Selezione territoriale"
)


region_names = (
    regioni[
        "regione"
    ]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)


regione_sel = st.sidebar.selectbox(
    "Regione",
    ["Italia"] + region_names,
)


provincia_sel = "Tutte"

if regione_sel != "Italia":

    prov_options = (
        province[
            province["regione"]
            .astype(str)
            == regione_sel
        ]["provincia"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    provincia_sel = (
        st.sidebar.selectbox(
            "Provincia / Città Metropolitana",
            ["Tutte"] + prov_options,
        )
    )


comune_sel = "Tutti"

if (
    regione_sel != "Italia"
    and provincia_sel != "Tutte"
):

    com_options = (
        comuni[
            (
                comuni["regione"]
                .astype(str)
                == regione_sel
            )
            &
            (
                comuni["provincia"]
                .astype(str)
                == provincia_sel
            )
        ]["comune"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    comune_sel = (
        st.sidebar.selectbox(
            "Comune",
            ["Tutti"] + com_options,
        )
    )



# ============================================================
# DATASET TERRITORIALE CORRENTE
# ============================================================

def dataset_corrente():

    if regione_sel == "Italia":

        return (
            regioni.copy(),
            "Regione",
        )


    if provincia_sel == "Tutte":

        df = province[
            province["regione"]
            .astype(str)
            == regione_sel
        ].copy()

        return (
            df,
            "Provincia",
        )


    if comune_sel == "Tutti":

        df = comuni[
            (
                comuni["regione"]
                .astype(str)
                == regione_sel
            )
            &
            (
                comuni["provincia"]
                .astype(str)
                == provincia_sel
            )
        ].copy()

        return (
            df,
            "Comune",
        )


    df = comuni[
        (
            comuni["regione"]
            .astype(str)
            == regione_sel
        )
        &
        (
            comuni["provincia"]
            .astype(str)
            == provincia_sel
        )
        &
        (
            comuni["comune"]
            .astype(str)
            == comune_sel
        )
    ].copy()

    return (
        df,
        "Cella 500 m",
    )


# ============================================================
# CONFINI ISTAT
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def estrai_limiti(
    zip_path_str,
):

    zip_path = Path(
        zip_path_str
    )

    out = (
        Path(
            tempfile.gettempdir()
        )
        / "sport_culture_limiti_2023"
    )

    marker = (
        out
        / "_estratto.txt"
    )

    if not marker.exists():

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as z:

            z.extractall(
                out
            )

        marker.write_text(
            "ok",
            encoding="utf-8",
        )

    return out



@st.cache_data(
    show_spinner=False,
    max_entries=1,
)
def carica_confini(
    livello,
):

    base = estrai_limiti(
        str(LIMITI_ZIP)
    )


    paths = {

        "regione":
            base
            / "Limiti01012023"
            / "Reg01012023"
            / "Reg01012023_WGS84.shp",

        "provincia":
            base
            / "Limiti01012023"
            / "ProvCM01012023"
            / "ProvCM01012023_WGS84.shp",

        "comune":
            base
            / "Limiti01012023"
            / "Com01012023"
            / "Com01012023_WGS84.shp",
    }


    gdf = gpd.read_file(
        paths[livello]
    )


    # --------------------------------------------------------
    # 1. Elimina geometrie assenti/vuote
    # --------------------------------------------------------

    gdf = gdf[
        gdf.geometry.notna()
    ].copy()

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()


    # --------------------------------------------------------
    # 2. WGS84 reale per Kepler
    # --------------------------------------------------------

    if gdf.crs is None:

        gdf = gdf.set_crs(
            "EPSG:4326",
            allow_override=True,
        )

    elif gdf.crs.to_epsg() != 4326:

        gdf = gdf.to_crs(
            "EPSG:4326"
        )


    # --------------------------------------------------------
    # 3. Controllo coordinate
    #
    # Kepler accetta:
    # longitude = -180 / +180
    # latitude  =  -90 / +90
    # --------------------------------------------------------

    bounds = gdf.geometry.bounds

    valid = (
        bounds["minx"].between(
            -180,
            180,
        )
        &
        bounds["maxx"].between(
            -180,
            180,
        )
        &
        bounds["miny"].between(
            -90,
            90,
        )
        &
        bounds["maxy"].between(
            -90,
            90,
        )
    )


    gdf = gdf[
        valid
    ].copy()


    return gdf

# ============================================================
# COLORI KEPLER
# ============================================================

def colori_valori(
    values,
):

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )


    valid = numeric.dropna()


    if valid.empty:

        return [
            [40, 165, 199]
            for _ in range(
                len(values)
            )
        ]


    minimo = valid.min()
    massimo = valid.max()


    if massimo == minimo:

        t = pd.Series(
            0.55,
            index=numeric.index,
        )

    else:

        t = (
            numeric - minimo
        ) / (
            massimo - minimo
        )


    t = (
        t
        .fillna(0.0)
        .clip(
            0,
            1,
        )
    )


    start = np.array(
        [
            194,
            240,
            246,
        ],
        dtype=float,
    )

    end = np.array(
        [
            8,
            74,
            104,
        ],
        dtype=float,
    )


    result = []

    for value in t:

        rgb = (
            start
            + value
            * (
                end - start
            )
        )

        result.append(
            [
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2]),
            ]
        )


    return result


# ============================================================
# PREPARAZIONE MAPPA AMMINISTRATIVA
# ============================================================

def prepara_mappa_admin(
    livello,
    df_indicatori,
    metrica,
):

    confini = carica_confini(
        livello
    )


    if livello == "regione":

        target_code = (
            "codice_regione_istat"
        )

        width = 2

        code_candidates = [
            "COD_REG",
            "COD_REGIONE",
        ]

        name_candidates = [
            "DEN_REG",
            "REGIONE",
        ]

        target_name = "regione"


    elif livello == "provincia":

        target_code = (
            "codice_provincia_istat"
        )

        width = 3

        code_candidates = [
            "COD_PROV",
            "COD_UTS",
        ]

        name_candidates = [
            "DEN_UTS",
            "DEN_PROV",
            "PROVINCIA",
        ]

        target_name = "provincia"


    else:

        target_code = (
            "codice_comune_istat"
        )

        width = 6

        code_candidates = [
            "PRO_COM_T",
            "PRO_COM",
            "COD_COM",
        ]

        name_candidates = [
            "COMUNE",
            "DEN_COM",
        ]

        target_name = "comune"


    boundary_code = trova_colonna(
        confini.columns,
        code_candidates,
    )


    if (
        boundary_code is not None
        and target_code
        in df_indicatori.columns
    ):

        confini[
            "_join"
        ] = normalizza_codice(
            confini[
                boundary_code
            ],
            width,
        )


        data = (
            df_indicatori.copy()
        )

        data[
            "_join"
        ] = normalizza_codice(
            data[
                target_code
            ],
            width,
        )


        merged = confini.merge(
            data,
            on="_join",
            how="inner",
        )


    else:

        boundary_name = trova_colonna(
            confini.columns,
            name_candidates,
        )


        if boundary_name is None:

            raise RuntimeError(
                "Impossibile individuare "
                f"il campo di join per {livello}."
            )


        confini[
            "_join_name"
        ] = (
            confini[
                boundary_name
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        data = (
            df_indicatori.copy()
        )

        data[
            "_join_name"
        ] = (
            data[
                target_name
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        merged = confini.merge(
            data,
            on="_join_name",
            how="inner",
        )


    merged[
        "valore_mappa"
    ] = pd.to_numeric(
        merged[
            metrica
        ],
        errors="coerce",
    )


    merged[
        "fillColor"
    ] = colori_valori(
        merged[
            "valore_mappa"
        ]
    )


    merged[
        "lineColor"
    ] = [
        [
            255,
            255,
            255,
        ]
        for _ in range(
            len(merged)
        )
    ]


    merged[
        "lineWidth"
    ] = 1


    if target_name in merged.columns:

        merged[
            "area"
        ] = merged[
            target_name
        ].astype(str)

    else:

        merged[
            "area"
        ] = ""


    keep = [
        "area",
        "valore_mappa",
        "fillColor",
        "lineColor",
        "lineWidth",
        "geometry",
    ]


    return merged[
        keep
    ].copy()


# ============================================================
# MAPPA 500 M PER COMUNE
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=1,
)
def carica_celle_comune(
    regione,
    provincia,
    comune,
    metrica,
):

    colonne = [
        "cell_id",
        "geometry",
        "regione",
        "provincia",
        "comune",
        metrica,
    ]


    dataset = pds.dataset(
        str(
            FILES[
                "contesto_500m"
            ]
        ),
        format="parquet",
    )


    disponibili = set(
        dataset.schema.names
    )


    mancanti = [
        c
        for c in colonne
        if c not in disponibili
    ]


    if mancanti:

        raise RuntimeError(
            "Colonne mancanti nel dataset 500 m: "
            + ", ".join(mancanti)
        )


    filtro = (
        (
            pds.field("regione")
            == regione
        )
        &
        (
            pds.field("provincia")
            == provincia
        )
        &
        (
            pds.field("comune")
            == comune
        )
    )


    table = dataset.to_table(
        columns=colonne,
        filter=filtro,
    )


    if table.num_rows == 0:

        return gpd.GeoDataFrame(
            columns=[
                "area",
                "valore_mappa",
                "fillColor",
                "lineColor",
                "lineWidth",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )


    df = table.to_pandas()


    # --------------------------------------------------------
    # GEOMETRIA
    # --------------------------------------------------------

    geom_raw = df[
        "geometry"
    ]


    valid_geom = (
        geom_raw
        .dropna()
    )


    if valid_geom.empty:

        return gpd.GeoDataFrame(
            columns=[
                "area",
                "valore_mappa",
                "fillColor",
                "lineColor",
                "lineWidth",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )


    sample = valid_geom.iloc[0]


    if isinstance(
        sample,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):

        geometry = (
            gpd.GeoSeries.from_wkb(
                geom_raw,
                crs="EPSG:3035",
            )
        )

    else:

        geometry = gpd.GeoSeries(
            geom_raw,
            crs="EPSG:3035",
        )


    df = df.drop(
        columns=[
            "geometry"
        ]
    )


    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:3035",
    )


    gdf = gdf.to_crs(
        "EPSG:4326"
    )


    # --------------------------------------------------------
    # CAMPI MAPPA
    # --------------------------------------------------------

    gdf[
        "area"
    ] = (
        gdf[
            "cell_id"
        ]
        .astype(str)
    )


    gdf[
        "valore_mappa"
    ] = pd.to_numeric(
        gdf[
            metrica
        ],
        errors="coerce",
    )


    gdf[
        "fillColor"
    ] = colori_valori(
        gdf[
            "valore_mappa"
        ]
    )


    gdf[
        "lineColor"
    ] = [
        [
            255,
            255,
            255,
        ]
        for _ in range(
            len(gdf)
        )
    ]


    gdf[
        "lineWidth"
    ] = 1


    return gdf[
        [
            "area",
            "valore_mappa",
            "fillColor",
            "lineColor",
            "lineWidth",
            "geometry",
        ]
    ].copy()


# ============================================================
# RENDER KEPLER
# ============================================================



def render_kepler(
    gdf,
    height=720,
):

    if gdf is None or gdf.empty:

        st.warning(
            "Nessun dato disponibile "
            "per la selezione corrente."
        )

        return


    # --------------------------------------------------------
    # GEOMETRIE VALIDE
    # --------------------------------------------------------

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()


    if gdf.empty:

        st.warning(
            "Nessuna geometria valida."
        )

        return


    # --------------------------------------------------------
    # CENTRO MAPPA
    # --------------------------------------------------------

    minx, miny, maxx, maxy = (
        gdf.total_bounds
    )


    longitude = (
        float(minx)
        + float(maxx)
    ) / 2


    latitude = (
        float(miny)
        + float(maxy)
    ) / 2


    span = max(
        float(maxx - minx),
        float(maxy - miny) * 1.35,
        0.01,
    )


    zoom = float(
        np.clip(
            math.log2(
                360.0 / span
            ) - 0.65,
            3.0,
            13.0,
        )
    )


    # --------------------------------------------------------
    # BASEMAP OPENFREEMAP
    #
    # Nessun token.
    # Nessuna API key.
    # --------------------------------------------------------

    if dark_mode:

        basemap_id = (
            "openfreemap_dark"
        )

        basemap_url = (
            "https://tiles.openfreemap.org/"
            "styles/dark"
        )

        basemap_label = (
            "OpenFreeMap Dark"
        )

    else:

        basemap_id = (
            "openfreemap_liberty"
        )

        basemap_url = (
            "https://tiles.openfreemap.org/"
            "styles/liberty"
        )

        basemap_label = (
            "OpenFreeMap Liberty"
        )


    # --------------------------------------------------------
    # CONFIG KEPLER
    # --------------------------------------------------------

    config = {
        "version": "v1",

        "config": {

            "mapState": {
                "latitude": latitude,
                "longitude": longitude,
                "zoom": zoom,
                "pitch": 0,
                "bearing": 0,
            },

            "mapStyle": {

                "styleType":
                    basemap_id,

                "topLayerGroups": {},

                "visibleLayerGroups": {
                    "label": True,
                    "road": True,
                    "border": True,
                    "building": True,
                    "water": True,
                    "land": True,
                    "3d building": False,
                },

                "mapStyles": {

                    basemap_id: {
                        "id":
                            basemap_id,

                        "label":
                            basemap_label,

                        "url":
                            basemap_url,
                    }
                },
            },
        },
    }


    # --------------------------------------------------------
    # DATI
    # --------------------------------------------------------

    geojson = json.loads(
        gdf.to_json(
            drop_id=True
        )
    )


    mappa = KeplerGl(
        height=height,

        data={
            "Territorio":
                geojson
        },

        config=config,
    )


    # --------------------------------------------------------
    # HTML TEMPORANEO
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".html",
        delete=False,
    ) as tmp:

        temp_path = Path(
            tmp.name
        )


    try:

        mappa.save_to_html(
            file_name=str(
                temp_path
            ),
            read_only=True,
        )


        html = temp_path.read_text(
            encoding="utf-8"
        )


        components.html(
            html,
            height=height,
            scrolling=False,
        )


    finally:

        try:

            temp_path.unlink(
                missing_ok=True
            )

        except Exception:

            pass


    del geojson
    del mappa

    try:
        del html
    except Exception:
        pass

    gc.collect()


def render_legenda_mappa(
    gdf,
    metric_label,
):

    if (
        gdf is None
        or gdf.empty
        or "valore_mappa"
        not in gdf.columns
    ):
        return


    valori = pd.to_numeric(
        gdf["valore_mappa"],
        errors="coerce",
    ).dropna()


    if valori.empty:
        return


    minimo = float(
        valori.min()
    )

    mediana = float(
        valori.median()
    )

    massimo = float(
        valori.max()
    )


    def formato(x):

        if abs(x) >= 1000:

            return (
                f"{x:,.1f}"
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )

        return (
            f"{x:.2f}"
            .replace(".", ",")
        )


    st.markdown(
        f"**Legenda — {metric_label}**"
    )


    st.markdown(
        """
        <div style="
            height:14px;
            border-radius:7px;
            margin-top:6px;
            margin-bottom:4px;
            background:
                linear-gradient(
                    90deg,
                    rgb(194,240,246) 0%,
                    rgb(8,74,104) 100%
                );
        ">
        </div>
        """,
        unsafe_allow_html=True,
    )


    c1, c2, c3 = st.columns(3)


    with c1:
        st.caption(
            f"Minimo: {formato(minimo)}"
        )


    with c2:
        st.caption(
            f"Mediana: {formato(mediana)}"
        )


    with c3:
        st.caption(
            f"Massimo: {formato(massimo)}"
        )


# ============================================================
# METRICHE MAPPA
# ============================================================

MAP_METRICS = {

    "Strutture per 10.000 abitanti":
        "strutture_per_10k",

    "Sport per 10.000 abitanti":
        "sport_per_10k",

    "Cultura per 10.000 abitanti":
        "cultura_per_10k",

    "Disponibilità locale":
        "disponibilita_locale_totale",

    "Concentrazione HHI":
        "concentrazione_hhi_totale",

    "Densità abitativa":
        "densita_abitativa_km2",
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "Sport & Culture Urban Climate"
)

st.write(
    "Sistema nazionale di analisi territoriale "
    "della fruibilità dell'offerta sportiva "
    "e culturale italiana."
)


# ============================================================
# KPI DASHBOARD - FONT
# ============================================================

st.markdown(
    """
<style>

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    font-weight: 700 !important;
}

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PANORAMA
# ============================================================

def page_panorama():

    st.header(
        "Panorama nazionale"
    )



    df, livello = dataset_corrente()


    popolazione = (
        df[
            "popolazione_2023"
        ]
        .sum()
    )


    # --------------------------------------------------------
    # OFFERTA
    #
    # Italia:
    # Master Registry completo.
    #
    # Regioni / Province / Comuni:
    # strutture territorialmente attribuite.
    # --------------------------------------------------------

    if regione_sel == "Italia":

        registry_dataset = pds.dataset(
            str(
                FILES["registry"]
            ),
            format="parquet",
        )


        strutture = (
            registry_dataset
            .count_rows()
        )


        sport = (
            registry_dataset
            .count_rows(
                filter=(
                    pds.field("dominio")
                    == "SPORT"
                )
            )
        )


        cultura = (
            registry_dataset
            .count_rows(
                filter=(
                    pds.field("dominio")
                    == "CULTURA"
                )
            )
        )


    else:

        strutture = (
            df[
                "n_strutture"
            ]
            .sum()
        )


        sport = (
            df[
                "n_sport"
            ]
            .sum()
        )


        cultura = (
            df[
                "n_cultura"
            ]
            .sum()
        )


    c1, c2, c3, c4 = (
        st.columns(
            4,
            gap="medium",
        )
    )


    c1.metric(
        "Popolazione 2023",
        fmt_int(
            popolazione
        ),
    )


    c2.metric(
        "Strutture",
        fmt_int(
            strutture
        ),
    )


    c3.metric(
        "Sport",
        fmt_int(
            sport
        ),
    )


    c4.metric(
        "Cultura",
        fmt_int(
            cultura
        ),
    )


    if regione_sel == "Italia":

        st.caption(
            "Il totale nazionale utilizza il Master Registry completo. "
            "753 strutture non dispongono di attribuzione "
            "regionale/provinciale e sono quindi incluse nel totale Italia "
            "ma escluse dalle aggregazioni territoriali."
        )


    st.subheader(
        "Mappa territoriale"
    )


    metric_label = (
        st.selectbox(
            "Indicatore da visualizzare",
            list(
                MAP_METRICS.keys()
            ),
        )
    )


    metrica = (
        MAP_METRICS[
            metric_label
        ]
    )


    with st.spinner(
        "Preparazione mappa Kepler..."
    ):

        if regione_sel == "Italia":

            gdf = prepara_mappa_admin(
                "regione",
                regioni,
                metrica,
            )


        elif provincia_sel == "Tutte":

            data = province[
                province["regione"]
                .astype(str)
                == regione_sel
            ].copy()


            gdf = prepara_mappa_admin(
                "provincia",
                data,
                metrica,
            )


        elif comune_sel == "Tutti":

            data = comuni[
                (
                    comuni["regione"]
                    .astype(str)
                    == regione_sel
                )
                &
                (
                    comuni["provincia"]
                    .astype(str)
                    == provincia_sel
                )
            ].copy()


            gdf = prepara_mappa_admin(
                "comune",
                data,
                metrica,
            )


        else:

            gdf = carica_celle_comune(
                regione_sel,
                provincia_sel,
                comune_sel,
                metrica,
            )


        render_kepler(
            gdf,
            height=760,
        )


    render_legenda_mappa(
        gdf,
        metric_label,
    )


    st.caption(
        "Italia → Regioni → Province/Città Metropolitane "
        "→ Comuni → griglia 500 m."
    )


    with st.expander(
        "Nota metodologica della mappa",
        expanded=False,
    ):

        st.write(
            "La mappa utilizza i confini amministrativi "
            "al 01/01/2023 presenti nel progetto. "
            "Alla scala nazionale vengono rappresentate "
            "le Regioni; selezionando una Regione si passa "
            "alle Province/Città Metropolitane; selezionando "
            "una Provincia si passa ai Comuni; selezionando "
            "un Comune la visualizzazione passa alla griglia "
            "analitica di 500 m."
        )

        st.write(
            "I colori rappresentano l'indicatore selezionato. "
            "La scala cromatica è calcolata linearmente tra "
            "il valore minimo e il valore massimo delle aree "
            "attualmente visualizzate. Questa operazione serve "
            "esclusivamente alla rappresentazione cartografica: "
            "non modifica i valori originali e non costituisce "
            "una nuova normalizzazione degli indicatori."
        )


    st.subheader(
        "Indicatori"
    )


    columns = [
        c
        for c in [
            "regione",
            "provincia",
            "comune",
            "popolazione_2023",
            "n_strutture",
            "n_sport",
            "n_cultura",
            "strutture_per_10k",
            "sport_per_10k",
            "cultura_per_10k",
            "densita_abitativa_km2",
            "concentrazione_hhi_totale",
            "disponibilita_locale_totale",
        ]
        if c in df.columns
    ]


    st.dataframe(
        df[
            columns
        ],
        width="stretch",
        hide_index=True,
    )


# ============================================================
# OFFERTA
# ============================================================

def page_offerta():

    st.header(
        "Offerta sportiva e culturale"
    )


    dataset = pds.dataset(
        str(
            FILES[
                "registry"
            ]
        ),
        format="parquet",
    )


    filtro = None


    def aggiungi_filtro(
        corrente,
        nuovo,
    ):

        if corrente is None:
            return nuovo

        return corrente & nuovo


    # --------------------------------------------------------
    # DOMINIO
    # --------------------------------------------------------

    if dominio == "Sport":

        filtro = aggiungi_filtro(
            filtro,
            (
                pds.field("dominio")
                == "SPORT"
            ),
        )


    elif dominio == "Cultura":

        filtro = aggiungi_filtro(
            filtro,
            (
                pds.field("dominio")
                == "CULTURA"
            ),
        )


    # --------------------------------------------------------
    # TERRITORIO
    # --------------------------------------------------------

    if regione_sel != "Italia":

        filtro = aggiungi_filtro(
            filtro,
            (
                pds.field("regione")
                == regione_sel
            ),
        )


    if provincia_sel != "Tutte":

        filtro = aggiungi_filtro(
            filtro,
            (
                pds.field("provincia")
                == provincia_sel
            ),
        )


    if comune_sel != "Tutti":

        filtro = aggiungi_filtro(
            filtro,
            (
                pds.field("comune")
                == comune_sel
            ),
        )


    # --------------------------------------------------------
    # NUMERO STRUTTURE
    # --------------------------------------------------------

    n_strutture = dataset.count_rows(
        filter=filtro
    )


    # --------------------------------------------------------
    # CATEGORIE E FONTI
    #
    # Scansione a batch:
    # non crea un DataFrame nazionale completo.
    # --------------------------------------------------------

    categorie = set()
    fonti = set()


    scanner = dataset.scanner(
        columns=[
            "categoria",
            "fonti",
        ],
        filter=filtro,
        batch_size=32768,
    )


    for batch in scanner.to_batches():

        dati = (
            batch.to_pydict()
        )


        categorie.update(
            x
            for x in dati[
                "categoria"
            ]
            if x is not None
        )


        fonti.update(
            x
            for x in dati[
                "fonti"
            ]
            if x is not None
        )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Strutture",
        fmt_int(
            n_strutture
        ),
    )


    c2.metric(
        "Categorie presenti",
        fmt_int(
            len(categorie)
        ),
    )


    c3.metric(
        "Fonti presenti",
        fmt_int(
            len(fonti)
        ),
    )


    # --------------------------------------------------------
    # SOLO PRIME 5.000 RIGHE
    # --------------------------------------------------------

    columns = [
        "master_id",
        "nome",
        "dominio",
        "categoria",
        "sottocategoria",
        "fonti",
        "n_fonti",
        "comune",
        "provincia",
        "regione",
    ]


    preview_table = dataset.head(
        5000,
        columns=columns,
        filter=filtro,
    )


    preview = (
        preview_table
        .to_pandas()
    )


    st.subheader(
        "Master Registry"
    )


    st.dataframe(
        preview,
        width="stretch",
        hide_index=True,
    )


    if n_strutture > 5000:

        st.caption(
            "La tabella mostra le prime "
            "5.000 righe su "
            f"{fmt_int(n_strutture)} strutture."
        )


    del preview
    del preview_table
    del scanner
    del categorie
    del fonti

    gc.collect()


# ============================================================
# POPOLAZIONE
# ============================================================

def page_popolazione():

    st.header(
        "Popolazione e territorio"
    )


    df, livello = dataset_corrente()


    popolazione = (
        df[
            "popolazione_2023"
        ]
        .sum()
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Popolazione 2023",
        fmt_int(
            popolazione
        ),
    )


    if (
        "densita_abitativa_km2"
        in df.columns
    ):

        c2.metric(
            "Densità abitativa media",
            fmt_float(
                df[
                    "densita_abitativa_km2"
                ]
                .mean(),
                1,
            ),
        )


    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# ACCESSIBILITÀ
# ============================================================


def page_accessibilita():

    st.header(
        "Accessibilità e fruibilità"
    )


    score = read_parquet(
        FILES[
            "score_accessibilita"
        ],
        columns=[
            "score_accessibilita"
        ],
    )


    buffer_df = read_parquet(
        FILES[
            "buffer_500m"
        ],
        columns=[
            "buffer_500m",
            "buffer_1km",
        ],
    )


    valori = (
        score[
            "score_accessibilita"
        ]
        .dropna()
    )


    c1, c2, c3 = (
        st.columns(
            3
        )
    )


    c1.metric(
        "Celle con score",
        fmt_int(
            len(valori)
        ),
    )


    c2.metric(
        "Score medio",
        fmt_float(
            valori.mean(),
            3,
        ),
    )


    c3.metric(
        "Score mediano",
        fmt_float(
            valori.median(),
            3,
        ),
    )


    st.subheader(
        "Buffer di accessibilità"
    )


    b1, b2 = st.columns(
        2
    )


    b1.metric(
        "Celle entro 500 m",
        fmt_int(
            (
                buffer_df[
                    "buffer_500m"
                ]
                == 1
            )
            .sum()
        ),
    )


    b2.metric(
        "Celle entro 1 km",
        fmt_int(
            (
                buffer_df[
                    "buffer_1km"
                ]
                == 1
            )
            .sum()
        ),
    )


    with st.container(
        border=True
    ):

        st.write(
            "Questa sezione utilizza "
            "gli output di accessibilità "
            "già prodotti nel progetto: "
            "score di accessibilità e "
            "buffer 500 m / 1 km."
        )


    del score
    del buffer_df
    del valori

    gc.collect()

# ============================================================
# CONTESTO E AMBIENTE
# ============================================================


def page_contesto():

    st.header(
        "Contesto urbano e ambiente"
    )


    # ========================================================
    # CONTESTO URBANO
    # ========================================================

    st.subheader(
        "Contesto urbano"
    )


    contesto_cols = [
        "cell_id",
        "regione",
        "provincia",
        "comune",
        "densita_abitativa_km2",
        "classe_territoriale",
        "tipologia_area",
        "share_sealed_pct",
        "share_built_up_pct",
        "tree_cover_density_pct",
        "n_complementary_total",
        "functional_hhi",
        "functional_mix",
        "is_urban_centre",
        "functional_intensity",
        "urban_tissue_continuity",
    ]


    contesto = read_parquet_preview(
        FILES[
            "contesto_500m"
        ],
        columns=contesto_cols,
        n_rows=5000,
    )


    st.dataframe(
        contesto,
        width="stretch",
        hide_index=True,
    )


    st.caption(
        "Anteprima delle prime 5.000 celle. "
        "Il dataset completo rimane invariato "
        "negli output del progetto."
    )


    # ========================================================
    # AMBIENTE
    # ========================================================

    st.subheader(
        "Ambiente"
    )


    a1, a2, a3 = st.columns(
        3
    )


    a1.metric(
        "Celle totali",
        "1.228.032",
    )


    a2.metric(
        "Celle senza alcun dato ambientale",
        "423",
    )


    a3.metric(
        "Quota completamente priva di dati",
        "0,03%",
    )


    st.info(
        "I valori mancanti non indicano necessariamente "
        "assenza di informazione ambientale. "
        "Solo 423 celle su 1.228.032 sono completamente "
        "prive di variabili ambientali. "
        "Per SVF, ombreggiamento, ore di sole diretto e "
        "proxy di esposizione solare il NoData è più esteso "
        "perché queste elaborazioni morfologiche sono state "
        "applicate nell'area analitica prevista attorno "
        "alle strutture; fuori da tale area il valore "
        "è stato mantenuto come NoData."
    )


    env_cols = [
        "cell_id",
        "svf_mean",
        "shaded_daylight_fraction_annual",
        "direct_sun_hours_annual",
        "solar_exposure_proxy_annual_kwh_m2",
        "ghi_annual_kwh_m2",
        "bhi_annual_kwh_m2",
        "dhi_annual_kwh_m2",
        "temperature_summer_degC",
        "relative_humidity_summer_pct",
        "utci_summer_degC",
        "thermal_comfort_pct_hours_summer",
    ]


    ambiente = read_parquet_preview(
        FILES[
            "ambiente_500m"
        ],
        columns=env_cols,
        n_rows=5000,
    )


    st.dataframe(
        ambiente,
        width="stretch",
        hide_index=True,
    )


    st.caption(
        "Anteprima delle prime 5.000 celle. "
        "I NoData sono mantenuti come tali: "
        "non vengono sostituiti con zero e non vengono imputati."
    )


    del contesto
    del ambiente

    gc.collect()


# ============================================================
# CONFRONTI
# ============================================================

def page_confronti():

    st.header(
        "Confronti territoriali"
    )


    ranking_regionale = read_parquet(
        FILES[
            "ranking_regionale"
        ]
    )


    ranking_provinciale = read_parquet(
        FILES[
            "ranking_provinciale"
        ]
    )


    def prepara_ranking_display(
        df
    ):

        out = df.copy()


        for col in [
            "popolazione_totale",
            "popolazione_con_indice",
        ]:

            if col in out.columns:

                out[col] = (
                    pd.to_numeric(
                        out[col],
                        errors="coerce",
                    )
                    .round(0)
                    .astype("Int64")
                )


        for col in [
            "copertura_popolazione_pct",
            "valore_normalizzato_regionale",
            "valore_normalizzato_provinciale",
        ]:

            if col in out.columns:

                out[col] = (
                    pd.to_numeric(
                        out[col],
                        errors="coerce",
                    )
                    .round(2)
                )


        return out


    ranking_regionale_display = (
        prepara_ranking_display(
            ranking_regionale
        )
    )


    ranking_provinciale_display = (
        prepara_ranking_display(
            ranking_provinciale
        )
    )


    st.caption(
        "Le variabili di popolazione sono visualizzate "
        "arrotondate alla persona più vicina. "
        "L'arrotondamento riguarda esclusivamente "
        "la visualizzazione della dashboard e non modifica "
        "gli output originali."
    )


    tab_reg, tab_prov = st.tabs(
        [
            "Regioni",
            "Province",
        ]
    )


    with tab_reg:

        st.dataframe(
            ranking_regionale_display,
            width="stretch",
            hide_index=True,
        )


    with tab_prov:

        st.dataframe(
            ranking_provinciale_display,
            width="stretch",
            hide_index=True,
        )


    del ranking_regionale
    del ranking_provinciale
    del ranking_regionale_display
    del ranking_provinciale_display

    gc.collect()


# ============================================================
# METODOLOGIA
# ============================================================

def page_metodologia():

    st.header(
        "Qualità dati e metodologia"
    )


    st.markdown(
        """
### Obiettivo

La dashboard supporta l'analisi della fruibilità
dell'offerta sportiva e culturale italiana mettendo
in relazione strutture, popolazione, territorio,
accessibilità, domanda/offerta e contesto urbano.

Il workflow generale seguito dal progetto è:

**dati → unità spaziali → analisi → indicatori →
normalizzazione → score → aggregazione →
visualizzazione.**
        """
    )


    # ========================================================
    # MASTER REGISTRY
    # ========================================================

    with st.expander(
        "1. Censimento nazionale dell'offerta",
        expanded=True,
    ):

        st.markdown(
            """
Il censimento integra fonti ufficiali, open data
e OpenStreetMap.

Il workflow applicato al registro delle strutture è:

**raccolta → pulizia → standardizzazione → matching →
deduplicazione → classificazione → validazione →
ID univoco.**

Il Master Registry attuale contiene **324.553
strutture univoche**:

- **248.954 SPORT**
- **75.599 CULTURA**

Le strutture sono classificate utilizzando la
tassonomia SPORT/CULTURA approvata dal progetto.

**753 strutture** non dispongono di attribuzione
regionale/provinciale. Sono mantenute nel Master
Registry nazionale ma non possono essere incluse
nelle aggregazioni amministrative.
            """
        )


    # ========================================================
    # POPOLAZIONE E TERRITORIO
    # ========================================================

    with st.expander(
        "2. Popolazione e unità territoriali",
        expanded=False,
    ):

        st.markdown(
            """
La domanda territoriale è rappresentata attraverso
la popolazione residente con riferimento al 2023.

La griglia di **500 m** costituisce l'unità analitica
principale utilizzata negli output finali della
dashboard.

I risultati sono successivamente aggregabili secondo:

**Cella → Comune → Provincia/Città Metropolitana →
Regione → Italia.**

I confini amministrativi utilizzati nella dashboard
sono quelli presenti nel progetto con riferimento
al **01/01/2023**.
            """
        )


    # ========================================================
    # OFFERTA / DOMANDA / ACCESSIBILITÀ
    # ========================================================

    with st.expander(
        "3. Offerta, domanda e accessibilità",
        expanded=False,
    ):

        st.markdown(
            """
Prima della costruzione degli score sono stati
mantenuti indicatori elementari interpretabili,
tra cui:

- numero e densità delle strutture;
- strutture per popolazione;
- distanze;
- copertura;
- disponibilità locale;
- concentrazione dell'offerta;
- prossimità;
- rapporto domanda/offerta.

La dashboard utilizza inoltre gli output di
accessibilità già prodotti nel progetto, compresi
i **buffer di 500 m e 1 km** e lo
**score di accessibilità**.

Gli indicatori elementari restano disponibili anche
quando vengono successivamente utilizzati nella
costruzione degli score.
            """
        )


    # ========================================================
    # CONTESTO URBANO
    # ========================================================

    with st.expander(
        "4. Contesto urbano",
        expanded=False,
    ):

        st.markdown(
            """
Il modulo di contesto urbano affianca agli indicatori
di offerta e popolazione informazioni territoriali
quali:

- densità abitativa;
- impermeabilizzazione e superficie costruita;
- copertura arborea;
- uso del suolo;
- servizi complementari;
- mix funzionale;
- centralità urbana;
- intensità funzionale;
- continuità del tessuto urbano.

Le variabili mantengono il proprio valore elementare
anche quando contribuiscono a indicatori o score
successivi.
            """
        )


    # ========================================================
    # AMBIENTE
    # ========================================================

    with st.expander(
        "5. Ambiente e comfort",
        expanded=False,
    ):

        st.markdown(
            """
Il modulo ambientale include gli output prodotti per:

- Sky View Factor;
- ombreggiamento;
- ore di sole diretto;
- proxy di esposizione solare;
- radiazione solare GHI/BHI/DHI;
- temperatura;
- umidità relativa;
- UTCI;
- comfort termico.

Per le elaborazioni morfologiche e solari di dettaglio
il **NoData è intenzionale fuori dall'area analitica
prevista** e non viene interpretato come valore zero.

L'audit del dataset ambientale finale mostra:

- **1.228.032 celle totali**;
- **423 celle completamente prive di dati ambientali**;
- quota completamente priva di dati: **0,03%**.
            """
        )


    # ========================================================
    # NORMALIZZAZIONE E SCORE
    # ========================================================

    with st.expander(
        "6. Normalizzazione, score e ranking",
        expanded=False,
    ):

        st.markdown(
            """
Gli indicatori con unità differenti non vengono
combinati direttamente.

Gli output del progetto distinguono quindi tra:

1. indicatori elementari;
2. variabili normalizzate;
3. score tematici;
4. aggregazioni territoriali e ranking.

Formule, componenti, regole sui missing e pesi
utilizzati negli output già prodotti sono conservati
nei relativi file di metadata e nel registro tecnico.

La dashboard non ricalcola né modifica tali valori:
li legge dagli output validati del progetto.
            """
        )


    # ========================================================
    # QUALITÀ
    # ========================================================

    with st.expander(
        "7. Qualità e copertura territoriale",
        expanded=False,
    ):

        q1, q2, q3 = st.columns(
            3
        )


        q1.metric(
            "Strutture senza Regione/Provincia",
            "753",
        )


        q2.metric(
            "Celle senza Regione/Provincia",
            "10.506",
        )


        q3.metric(
            "Celle con dati ambientali totalmente assenti",
            "423",
        )


        st.markdown(
            """
L'audit territoriale della griglia 500 m ha inoltre
evidenziato:

- **10.506 celle** senza Regione e Provincia;
- tutte queste 10.506 celle hanno
  **densità abitativa pari a zero**;
- **95 celle** presentano densità positiva ma non
  dispongono dell'attribuzione comunale completa.

Queste 95 celle vengono mantenute nei dati originali
e segnalate come elemento da verificare, senza
attribuzioni o imputazioni artificiali.

La scarsità o l'assenza di un valore nel dataset non
viene quindi automaticamente interpretata come
assenza reale del fenomeno.
            """
        )


    # ========================================================
    # REGISTRO TECNICO
    # ========================================================

    registro = read_parquet(
        FILES[
            "registro_tecnico"
        ]
    )


    st.subheader(
        "Registro tecnico delle variabili"
    )


    st.dataframe(
        registro,
        width="stretch",
        hide_index=True,
    )


    # ========================================================
    # TASSONOMIA
    # ========================================================

    tassonomia = read_json(
        FILES[
            "tassonomia"
        ]
    )


    st.subheader(
        "Tassonomia approvata"
    )


    for dominio_tax in [
        "SPORT",
        "CULTURA",
    ]:

        with st.expander(
            dominio_tax,
            expanded=False,
        ):

            for (
                codice,
                label,
            ) in tassonomia[
                dominio_tax
            ].items():

                st.write(
                    f"**{label}** — `{codice}`"
                )


    del registro

    gc.collect()


# ============================================================
# ROUTER
# ============================================================

if pagina == "Panorama nazionale":

    page_panorama()


elif pagina == "Offerta sportiva e culturale":

    page_offerta()


elif pagina == "Popolazione e territorio":

    page_popolazione()


elif pagina == "Accessibilità e fruibilità":

    page_accessibilita()


elif pagina == "Contesto urbano e ambiente":

    page_contesto()


elif pagina == "Confronti territoriali":

    page_confronti()


elif pagina == "Qualità dati e metodologia":

    page_metodologia()


st.markdown("---")

st.caption(
    "Sport & Culture Urban Climate · Italia"
)

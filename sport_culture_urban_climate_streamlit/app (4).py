from pathlib import Path
import json
import tempfile
import zipfile
import gc
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
    show_spinner=False
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
    max_entries=3,
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


    try:

        gdf = gpd.read_parquet(
            FILES[
                "contesto_500m"
            ],
            columns=colonne,
            filters=[
                (
                    "regione",
                    "==",
                    regione,
                ),
                (
                    "provincia",
                    "==",
                    provincia,
                ),
                (
                    "comune",
                    "==",
                    comune,
                ),
            ],
        )


    except Exception:

        gdf = gpd.read_parquet(
            FILES[
                "contesto_500m"
            ],
            columns=colonne,
        )


        gdf = gdf[
            (
                gdf["regione"]
                .astype(str)
                == regione
            )
            &
            (
                gdf["provincia"]
                .astype(str)
                == provincia
            )
            &
            (
                gdf["comune"]
                .astype(str)
                == comune
            )
        ].copy()


    if gdf.crs is None:

        gdf = gdf.set_crs(
            3035
        )


    gdf = gdf.to_crs(
        4326
    )


    gdf[
        "area"
    ] = gdf[
        "cell_id"
    ].astype(str)


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

    if gdf.empty:

        st.warning(
            "Nessun dato disponibile "
            "per la selezione corrente."
        )

        return


    # --------------------------------------------------------
    # ELIMINA GEOMETRIE VUOTE
    # --------------------------------------------------------

    gdf = gdf[
        gdf.geometry.notna()
    ].copy()

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()


    if gdf.empty:

        st.warning(
            "Nessuna geometria valida "
            "per la selezione corrente."
        )

        return


    # --------------------------------------------------------
    # VISTA INIZIALE
    # calcolata sul territorio mostrato
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


    span_x = max(
        float(maxx - minx),
        0.01,
    )

    span_y = max(
        float(maxy - miny),
        0.01,
    )

    span = max(
        span_x,
        span_y * 1.4,
    )


    zoom = (
        math.log2(
            360.0 / span
        )
        - 0.9
    )

    zoom = float(
        np.clip(
            zoom,
            3.0,
            12.0,
        )
    )


    # --------------------------------------------------------
    # GEOJSON
    # --------------------------------------------------------

    geojson = json.loads(
        gdf.to_json(
            drop_id=True
        )
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
                    "dark"
                    if dark_mode
                    else "light",

                "visibleLayerGroups": {
                    "label": True,
                    "road": True,
                    "border": True,
                    "building": True,
                    "water": True,
                    "land": True,
                },
            },
        },
    }

    # --------------------------------------------------------
    # CREA UNA SOLA MAPPA
    # --------------------------------------------------------

    mappa = KeplerGl(
        height=height,
        data={
            "Territorio":
                geojson
        },
        config=config,
    )


    # --------------------------------------------------------
    # FILE TEMPORANEO UNICO
    #
    # Evita collisioni quando più utenti
    # aprono la dashboard contemporaneamente.
    # --------------------------------------------------------

    temp = tempfile.NamedTemporaryFile(
        prefix="sport_culture_kepler_",
        suffix=".html",
        delete=False,
    )

    temp_path = Path(
        temp.name
    )

    temp.close()


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


        # Streamlit recente
        if hasattr(
            st,
            "iframe"
        ):

            st.iframe(
                html,
                width="stretch",
                height=height,
            )

        # fallback versioni precedenti
        else:

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


    # --------------------------------------------------------
    # LIBERA MEMORIA
    # --------------------------------------------------------

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
        or "valore_mappa" not in gdf.columns
    ):
        return

    valori = pd.to_numeric(
        gdf["valore_mappa"],
        errors="coerce",
    ).dropna()

    if valori.empty:
        return

    minimo = float(valori.min())
    mediana = float(valori.median())
    massimo = float(valori.max())

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


    strutture = (
        df[
            "n_strutture"
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

def carica_registry():

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


    return pd.read_parquet(
        FILES["registry"],
        columns=columns,
    )


def page_offerta():

    st.header(
        "Offerta sportiva e culturale"
    )


    df = (
        carica_registry()
        .copy()
    )


    if dominio == "Sport":

        df = df[
            df["dominio"]
            .astype(str)
            .str.upper()
            == "SPORT"
        ]


    elif dominio == "Cultura":

        df = df[
            df["dominio"]
            .astype(str)
            .str.upper()
            == "CULTURA"
        ]


    if regione_sel != "Italia":

        df = df[
            df["regione"]
            .astype(str)
            == regione_sel
        ]


    if provincia_sel != "Tutte":

        df = df[
            df["provincia"]
            .astype(str)
            == provincia_sel
        ]


    if comune_sel != "Tutti":

        df = df[
            df["comune"]
            .astype(str)
            == comune_sel
        ]


    c1, c2, c3 = (
        st.columns(
            3
        )
    )


    c1.metric(
        "Strutture",
        fmt_int(
            len(df)
        ),
    )


    c2.metric(
        "Categorie presenti",
        fmt_int(
            df[
                "categoria"
            ]
            .nunique()
        ),
    )


    c3.metric(
        "Fonti presenti",
        fmt_int(
            df[
                "fonti"
            ]
            .nunique()
        ),
    )


    st.subheader(
        "Master Registry"
    )


    st.dataframe(
        df.head(
            5000
        ),
        width="stretch",
        hide_index=True,
    )


    if len(df) > 5000:

        st.caption(
            "La tabella mostra le prime 5.000 righe "
            f"su {fmt_int(len(df))} strutture."
        )


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
        "Il dataset originale completo rimane "
        "disponibile nel progetto."
    )


    # ========================================================
    # AMBIENTE
    # ========================================================

    st.subheader(
        "Ambiente"
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
        "Nessuna variabile viene eliminata "
        "dal dataset originale."
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


    ranking_regionale = (
        read_parquet(
            FILES[
                "ranking_regionale"
            ]
        )
    )


    ranking_provinciale = (
        read_parquet(
            FILES[
                "ranking_provinciale"
            ]
        )
    )


    tab_reg, tab_prov = st.tabs(
        [
            "Regioni",
            "Province",
        ]
    )


    with tab_reg:

        st.dataframe(
            ranking_regionale,
            width="stretch",
            hide_index=True,
        )


    with tab_prov:

        st.dataframe(
            ranking_provinciale,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# METODOLOGIA
# ============================================================

def page_metodologia():

    st.header(
        "Qualità dati e metodologia"
    )


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
            expanded=True,
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

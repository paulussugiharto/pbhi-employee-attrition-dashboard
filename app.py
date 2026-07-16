from datetime import datetime
from io import BytesIO
from pathlib import Path
import hashlib
import json

import csv
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Early Warning System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "attrition_model_bundle.joblib"
)

TEMPLATE_PATH = (
    BASE_DIR
    / "attrition_upload_template.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "model_metadata.json"
)


# ============================================================
# LOAD MODEL ASSETS
# ============================================================

@st.cache_resource
def load_bundle():

    return joblib.load(
        MODEL_PATH
    )


@st.cache_data
def load_template():

    return pd.read_csv(
        TEMPLATE_PATH
    )


@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():

        return {}

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as metadata_file:

        return json.load(
            metadata_file
        )


try:

    bundle = load_bundle()

    template_df = (
        load_template()
    )

    metadata = {
        **bundle.get(
            "metadata",
            {},
        ),

        **load_metadata(),
    }

except Exception as error:

    st.error(
        "Model, template, atau metadata gagal dimuat."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

pipeline = bundle[
    "pipeline"
]

model_name = metadata.get(
    "model_name",
    "CatBoost",
)

selected_features = list(
    bundle.get(
        "selected_features",

        metadata.get(
            "selected_features",
            [],
        ),
    )
)

required_columns = list(
    bundle.get(
        "raw_required_columns",

        metadata.get(
            "raw_required_columns",
            [],
        ),
    )
)

prediction_threshold = float(
    metadata.get(
        "prediction_threshold",

        bundle.get(
            "prediction_threshold",
            0.18,
        ),
    )
)

low_medium_cutoff = float(
    metadata.get(
        "low_medium_cutoff",
        0.09,
    )
)

medium_high_cutoff = float(
    metadata.get(
        "medium_high_cutoff",
        prediction_threshold,
    )
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "results": None,
    "uploaded_data": None,
    "file_hash": None,
    "filename": None,
    "processed_at": None,
    "quality": None,
    "intervention_status": {},
}


for (
    key,
    default_value,
) in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = default_value


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality(
    dataframe,
):

    normalized_columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    missing_required = [
        column
        for column in required_columns
        if column
        not in normalized_columns
    ]

    duplicate_ids = 0

    if (
        "EmployeeNumber"
        in dataframe.columns
    ):

        duplicate_ids = int(
            dataframe[
                "EmployeeNumber"
            ]
            .duplicated()
            .sum()
        )

    return {
        "rows":
            len(dataframe),

        "columns":
            len(
                dataframe.columns
            ),

        "required_available":
            (
                len(
                    required_columns
                )
                - len(
                    missing_required
                )
            ),

        "required_total":
            len(
                required_columns
            ),

        "missing_required":
            missing_required,

        "missing_cells":
            int(
                dataframe
                .isna()
                .sum()
                .sum()
            ),

        "duplicate_ids":
            duplicate_ids,
    }


# ============================================================
# PREPROCESSING AND PREDICTION
# ============================================================

def prepare_and_predict(
    uploaded_df,
):

    if uploaded_df.empty:

        raise ValueError(
            "File tidak memiliki baris data karyawan."
        )

    data = (
        uploaded_df.copy()
    )

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    duplicate_columns = (
        data.columns[
            data.columns
            .duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            "Nama kolom duplikat: "
            + ", ".join(
                duplicate_columns
            )
        )

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Kolom wajib berikut belum tersedia:\n"
            + "\n".join(
                missing_columns
            )
        )

    invalid_numeric = []

    for column in required_columns:

        if column not in template_df.columns:

            continue

        if pd.api.types.is_numeric_dtype(
            template_df[
                column
            ]
        ):

            original_values = (
                data[
                    column
                ]
                .copy()
            )

            data[
                column
            ] = pd.to_numeric(
                data[
                    column
                ],
                errors="coerce",
            )

            invalid_values = (
                data[
                    column
                ]
                .isna()

                & original_values
                .notna()
            )

            if invalid_values.any():

                invalid_numeric.append(
                    column
                )

        else:

            data[
                column
            ] = (
                data[
                    column
                ]
                .astype(
                    "string"
                )
                .str.strip()
            )

    if invalid_numeric:

        raise ValueError(
            "Kolom berikut seharusnya berisi angka:\n"
            + "\n".join(
                invalid_numeric
            )
        )

    missing_values = [
        column
        for column
        in required_columns
        if data[
            column
        ].isna().any()
    ]

    if missing_values:

        raise ValueError(
            "Terdapat nilai kosong pada kolom wajib:\n"
            + "\n".join(
                missing_values
            )
        )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    overtime = (
        data[
            "OverTime"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    stock_option = (
        pd.to_numeric(
            data[
                "StockOptionLevel"
            ],
            errors="coerce",
        )
    )

    years_company = (
        pd.to_numeric(
            data[
                "YearsAtCompany"
            ],
            errors="coerce",
        )
    )

    data[
        "OverTime_NoStockOption"
    ] = (
        overtime.eq(
            "yes"
        )

        & stock_option.eq(
            0
        )
    ).astype(int)

    data[
        "OverTime_ShortTenure"
    ] = (
        overtime.eq(
            "yes"
        )

        & years_company.le(
            2
        )
    ).astype(int)

    missing_features = [
        feature
        for feature
        in selected_features
        if feature
        not in data.columns
    ]

    if missing_features:

        raise ValueError(
            "Fitur model belum lengkap:\n"
            + "\n".join(
                missing_features
            )
        )

    X_prediction = (
        data[
            selected_features
        ]
        .copy()
    )

    probability = (
        pipeline
        .predict_proba(
            X_prediction
        )[:, 1]
    )

    predicted_attrition = (
        np.where(
            probability
            >= prediction_threshold,

            "Yes",
            "No",
        )
    )

    risk_level = (
        np.select(
            [
                probability
                < low_medium_cutoff,

                probability
                < medium_high_cutoff,
            ],

            [
                "Low Risk",
                "Medium Risk",
            ],

            default=(
                "High Risk"
            ),
        )
    )

    recommended_action = (
        np.select(
            [
                risk_level
                == "High Risk",

                risk_level
                == "Medium Risk",
            ],

            [
                "Prioritize HR review",
                "Monitor and follow up",
            ],

            default=(
                "Standard monitoring"
            ),
        )
    )

    results = (
        data.drop(
            columns=[
                "OverTime_NoStockOption",
                "OverTime_ShortTenure",
            ],
            errors="ignore",
        )
    )

    if (
        "EmployeeNumber"
        not in results.columns
    ):

        results[
            "EmployeeNumber"
        ] = np.arange(
            1,
            len(
                results
            ) + 1,
        )

    results[
        "EmployeeNumber"
    ] = (
        results[
            "EmployeeNumber"
        ]
        .astype(str)
        .str.strip()
    )

    results[
        "Attrition_Probability"
    ] = probability

    results[
        "Probability_Percent"
    ] = (
        probability
        * 100
    ).round(2)

    results[
        "Predicted_Attrition"
    ] = predicted_attrition

    results[
        "Risk_Level"
    ] = risk_level

    results[
        "Recommended_Action"
    ] = recommended_action

    results = (
        results
        .sort_values(
            "Attrition_Probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results.insert(
        0,
        "Priority_Rank",
        range(
            1,
            len(
                results
            ) + 1,
        ),
    )

    return results


# ============================================================
# SUMMARY
# ============================================================

def get_summary(
    results,
):

    total = len(
        results
    )

    high = int(
        (
            results[
                "Risk_Level"
            ]
            == "High Risk"
        ).sum()
    )

    medium = int(
        (
            results[
                "Risk_Level"
            ]
            == "Medium Risk"
        ).sum()
    )

    low = int(
        (
            results[
                "Risk_Level"
            ]
            == "Low Risk"
        ).sum()
    )

    return {
        "total":
            total,

        "high":
            high,

        "medium":
            medium,

        "low":
            low,

        "monitoring":
            high
            + medium,

        "average":
            float(
                results[
                    "Attrition_Probability"
                ]
                .mean()
            ),
    }


# ============================================================
# FILTERING
# ============================================================

def filter_results(
    results,
    job_level,
    job_role,
    overtime,
):

    filtered = (
        results.copy()
    )

    if job_level != "All":

        filtered = filtered[
            pd.to_numeric(
                filtered[
                    "JobLevel"
                ],
                errors="coerce",
            )
            == float(
                job_level
            )
        ]

    if job_role != "All":

        filtered = filtered[
            filtered[
                "JobRole"
            ]
            == job_role
        ]

    if overtime != "All":

        filtered = filtered[
            filtered[
                "OverTime"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == overtime.lower()
        ]

    return filtered


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(
    label,
    value,
    note,
    accent,
    icon,
):

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-icon {accent}">
                {icon}
            </div>

            <div class="metric-content">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

                <div class="metric-note">
                    {note}
                </div>

            </div>

        </div>
        """
    )


def prediction_table(
    results,
    limit=None,
    height=None,
):

    if limit:

        table = (
            results
            .head(
                limit
            )
            .copy()
        )

    else:

        table = (
            results.copy()
        )

    table[
        "Risk_Display"
    ] = (
        table[
            "Risk_Level"
        ]
        .map(
            {
                "High Risk":
                    "🔴 High Risk",

                "Medium Risk":
                    "🟠 Medium Risk",

                "Low Risk":
                    "🟢 Low Risk",
            }
        )
    )

    columns = [
        "Priority_Rank",
        "EmployeeNumber",
        "JobRole",
        "JobLevel",
        "Probability_Percent",
        "Risk_Display",
        "Predicted_Attrition",
        "Recommended_Action",
    ]

    columns = [
        column
        for column
        in columns
        if column
        in table.columns
    ]

    st.dataframe(
        table[
            columns
        ],

        use_container_width=True,

        hide_index=True,

        height=(
            height

            or (
                390
                if limit
                else 560
            )
        ),

        column_config={

            "Priority_Rank":
                st.column_config.NumberColumn(
                    "Priority",
                    format="%d",
                ),

            "EmployeeNumber":
                st.column_config.TextColumn(
                    "Employee ID"
                ),

            "JobRole":
                st.column_config.TextColumn(
                    "Job Role"
                ),

            "JobLevel":
                st.column_config.NumberColumn(
                    "Job Level",
                    format="%d",
                ),

            "Probability_Percent":
                st.column_config.ProgressColumn(
                    "Risk Score",

                    min_value=0.0,

                    max_value=100.0,

                    format="%.2f%%",
                ),

            "Risk_Display":
                st.column_config.TextColumn(
                    "Risk Level"
                ),

            "Predicted_Attrition":
                st.column_config.TextColumn(
                    "Predicted Attrition"
                ),

            "Recommended_Action":
                st.column_config.TextColumn(
                    "Recommended Action"
                ),
        },
    )


def download_button(
    results,
    filename,
    label=(
        "⬇️ Download Results"
    ),
):

    st.download_button(
        label,

        data=(
            results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        ),

        file_name=(
            filename
        ),

        mime="text/csv",

        use_container_width=True,
    )


def clean_plot(
    figure,
    height,
):

    figure.update_layout(
        height=height,

        margin=dict(
            l=10,
            r=10,
            t=18,
            b=10,
        ),

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        font=dict(
            family="Arial",
            color="#1f344d",
        ),

        legend_title_text="",
    )


# ============================================================
# UPLOAD
# ============================================================

def upload_component():

    uploaded_file = (
        st.file_uploader(
            "Upload employee CSV",

            type=[
                "csv"
            ],

            label_visibility=(
                "collapsed"
            ),

            key=(
                "employee_file_uploader"
            ),
        )
    )

    st.download_button(
        "⬇️ Download CSV Template",

        data=(
            template_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        ),

        file_name=(
            "attrition_upload_template.csv"
        ),

        mime="text/csv",

        use_container_width=True,
    )

    if uploaded_file is None:

        st.info(
            "Download template, isi data karyawan, "
            "lalu upload kembali."
        )

        return

    try:

        # ====================================================
        # READ FILE
        # ====================================================

        file_bytes = (
            uploaded_file
            .getvalue()
        )

        if not file_bytes:

            raise ValueError(
                "File CSV kosong."
            )

        current_hash = (
            hashlib.md5(
                file_bytes
            )
            .hexdigest()
        )

        # ====================================================
        # VALIDATE RAW CSV HEADER
        # Harus dilakukan sebelum pd.read_csv(),
        # karena pandas dapat mengganti nama kolom duplikat.
        # ====================================================

        try:

            decoded_file = (
                file_bytes
                .decode(
                    "utf-8-sig"
                )
            )

        except UnicodeDecodeError:

            raise ValueError(
                "File tidak menggunakan encoding UTF-8 "
                "atau format file tidak valid."
            )

        file_lines = (
            decoded_file
            .splitlines()
        )

        if not file_lines:

            raise ValueError(
                "File CSV tidak memiliki header."
            )

        raw_columns = next(
            csv.reader(
                [
                    file_lines[0]
                ]
            )
        )

        raw_columns = [
            str(column).strip()
            for column
            in raw_columns
        ]

        if not raw_columns:

            raise ValueError(
                "File CSV tidak memiliki nama kolom."
            )

        empty_column_names = [
            index + 1
            for index, column
            in enumerate(
                raw_columns
            )
            if not column
        ]

        if empty_column_names:

            raise ValueError(
                "Terdapat nama kolom kosong pada posisi:\n"
                + "\n".join(
                    str(position)
                    for position
                    in empty_column_names
                )
            )

        duplicate_columns = sorted(
            {
                column
                for column
                in raw_columns
                if raw_columns.count(
                    column
                ) > 1
            }
        )

        if duplicate_columns:

            raise ValueError(
                "Nama kolom duplikat ditemukan:\n"
                + "\n".join(
                    duplicate_columns
                )
            )

        # ====================================================
        # READ CSV WITH PANDAS
        # ====================================================

        uploaded_df = (
            pd.read_csv(
                BytesIO(
                    file_bytes
                )
            )
        )

        uploaded_df.columns = (
            uploaded_df.columns
            .astype(str)
            .str.strip()
        )

        if uploaded_df.empty:

            raise ValueError(
                "File tidak memiliki baris data karyawan."
            )

        # ====================================================
        # VALIDATE EMPLOYEE ID
        # ====================================================

        if (
            "EmployeeNumber"
            in uploaded_df.columns
        ):

            employee_ids = (
                uploaded_df[
                    "EmployeeNumber"
                ]
                .astype(
                    "string"
                )
                .str.strip()
            )

            missing_employee_id = (
                employee_ids
                .isna()

                | employee_ids
                .eq("")
            )

            if missing_employee_id.any():

                missing_rows = (
                    uploaded_df.index[
                        missing_employee_id
                    ]
                    + 2
                ).tolist()

                raise ValueError(
                    "Employee ID kosong ditemukan "
                    "pada baris CSV:\n"
                    + "\n".join(
                        str(row)
                        for row
                        in missing_rows
                    )
                )

            duplicate_employee_mask = (
                employee_ids
                .duplicated(
                    keep=False
                )
            )

            if duplicate_employee_mask.any():

                duplicate_employee_ids = (
                    employee_ids[
                        duplicate_employee_mask
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

                duplicate_details = []

                for employee_id in (
                    duplicate_employee_ids
                ):

                    duplicate_rows = (
                        uploaded_df.index[
                            employee_ids
                            .eq(
                                employee_id
                            )
                        ]
                        + 2
                    ).tolist()

                    duplicate_details.append(
                        f"{employee_id} "
                        f"(baris: "
                        f"{', '.join(map(str, duplicate_rows))})"
                    )

                raise ValueError(
                    "Employee ID duplikat ditemukan:\n"
                    + "\n".join(
                        duplicate_details
                    )
                )

        # ====================================================
        # PROCESS NEW FILE
        # ====================================================

        if (
            current_hash
            != st.session_state[
                "file_hash"
            ]
        ):

            with st.spinner(
                "Model sedang memproses data karyawan..."
            ):

                results = (
                    prepare_and_predict(
                        uploaded_df
                    )
                )

            st.session_state[
                "results"
            ] = results

            st.session_state[
                "uploaded_data"
            ] = uploaded_df

            st.session_state[
                "file_hash"
            ] = current_hash

            st.session_state[
                "filename"
            ] = uploaded_file.name

            st.session_state[
                "processed_at"
            ] = (
                datetime.now()
                .strftime(
                    "%d %b %Y, %H:%M"
                )
            )

            st.session_state[
                "quality"
            ] = (
                data_quality(
                    uploaded_df
                )
            )

            st.rerun()

        # ====================================================
        # SUCCESS INFORMATION
        # ====================================================

        st.success(
            f"File aktif: "
            f"{uploaded_file.name} — "
            f"{len(uploaded_df):,} karyawan."
        )

        with st.expander(
            "Preview Uploaded Data"
        ):

            st.dataframe(
                uploaded_df
                .head(10),

                use_container_width=True,

                hide_index=True,
            )

    except Exception as error:

        for (
            key,
            default_value,
        ) in DEFAULT_STATE.items():

            st.session_state[
                key
            ] = default_value

        st.error(
            "File tidak dapat diproses."
        )

        st.code(
            str(error)
        )


# ============================================================
# CHART DATA
# ============================================================

def risk_distribution(
    results,
):

    return (
        results[
            "Risk_Level"
        ]
        .value_counts()
        .reindex(
            [
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ],

            fill_value=0,
        )
        .rename_axis(
            "Risk Level"
        )
        .reset_index(
            name="Employees"
        )
    )


def job_role_summary(
    results,
):

    grouped = (
        results
        .groupby(
            "JobRole"
        )
        .agg(

            Employees=(
                "EmployeeNumber",
                "size",
            ),

            High_Risk_Employees=(
                "Risk_Level",

                lambda values: (
                    values
                    == "High Risk"
                ).sum(),
            ),

            Average_Risk=(
                "Attrition_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped[
        "High Risk Rate"
    ] = (
        grouped[
            "High_Risk_Employees"
        ]
        / grouped[
            "Employees"
        ]
        * 100
    )

    grouped[
        "Average Risk Score"
    ] = (
        grouped[
            "Average_Risk"
        ]
        * 100
    )

    grouped[
        "Role Label"
    ] = (
        grouped[
            "JobRole"
        ]
        + " (n="
        + grouped[
            "Employees"
        ]
        .astype(str)
        + ")"
    )

    return (
        grouped
        .sort_values(
            [
                "High Risk Rate",
                "Average Risk Score",
            ],

            ascending=False,
        )
    )


def job_level_summary(
    results,
):

    grouped = (
        results
        .groupby(
            "JobLevel"
        )
        .agg(

            Employees=(
                "EmployeeNumber",
                "size",
            ),

            Average_Risk=(
                "Attrition_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped[
        "Average Risk Score"
    ] = (
        grouped[
            "Average_Risk"
        ]
        * 100
    )

    return (
        grouped
        .sort_values(
            "JobLevel"
        )
    )


# ============================================================
# RISK SIGNALS
# ============================================================

def risk_signal_data(
    results,
):

    high_risk = (
        results[
            results[
                "Risk_Level"
            ]
            == "High Risk"
        ]
        .copy()
    )

    if high_risk.empty:

        base = (
            results.copy()
        )

    else:

        base = high_risk

    base_count = len(
        base
    )

    income_median = (
        pd.to_numeric(
            results[
                "MonthlyIncome"
            ],
            errors="coerce",
        )
        .median()
    )

    definitions = [

        (
            "OverTime = Yes",

            "Employees working overtime.",

            base[
                "OverTime"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes"),

            "◷",
        ),

        (
            "Job Level = 1",

            "Entry-level employee profile.",

            pd.to_numeric(
                base[
                    "JobLevel"
                ],
                errors="coerce",
            )
            .eq(1),

            "▣",
        ),

        (
            "Income below median",

            "Monthly income below selected-data median.",

            pd.to_numeric(
                base[
                    "MonthlyIncome"
                ],
                errors="coerce",
            )
            < income_median,

            "$",
        ),

        (
            "Years at Company ≤ 2",

            "Short organizational tenure.",

            pd.to_numeric(
                base[
                    "YearsAtCompany"
                ],
                errors="coerce",
            )
            .le(2),

            "⌛",
        ),

        (
            "Stock Option Level = 0",

            "Employees without stock options.",

            pd.to_numeric(
                base[
                    "StockOptionLevel"
                ],
                errors="coerce",
            )
            .eq(0),

            "◇",
        ),
    ]

    rows = []

    for (
        name,
        description,
        mask,
        icon,
    ) in definitions:

        count = int(
            mask
            .fillna(False)
            .sum()
        )

        if base_count:

            rate = (
                count
                / base_count
                * 100
            )

        else:

            rate = 0

        rows.append(
            {
                "Signal":
                    name,

                "Description":
                    description,

                "Count":
                    count,

                "Rate":
                    rate,

                "Icon":
                    icon,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "Rate",
            ascending=False,
        )
    )


def signal_panel(
    results,
):

    signal_data = (
        risk_signal_data(
            results
        )
    )

    for (
        _,
        row,
    ) in signal_data.iterrows():

        if row[
            "Rate"
        ] >= 60:

            css_class = (
                "signal-high"
            )

        elif row[
            "Rate"
        ] >= 30:

            css_class = (
                "signal-medium"
            )

        else:

            css_class = (
                "signal-low"
            )

        st.html(
            f"""
            <div class="signal-row">

                <div class="signal-icon">
                    {row["Icon"]}
                </div>

                <div class="signal-main">

                    <b>
                        {row["Signal"]}
                    </b>

                    <small>
                        {row["Description"]}
                    </small>

                </div>

                <div class="{css_class}">

                    <b>
                        {row["Rate"]:.1f}%
                    </b>

                    <small>
                        {int(row["Count"])}
                        employees
                    </small>

                </div>

            </div>
            """
        )


# ============================================================
# EMPLOYEE DETAIL
# ============================================================

def employee_detail(
    results,
):

    selected_id = (
        st.selectbox(
            "Select Employee ID",

            results[
                "EmployeeNumber"
            ]
            .astype(str)
            .tolist(),

            key=(
                "employee_detail_id"
            ),
        )
    )

    employee = (
        results[
            results[
                "EmployeeNumber"
            ]
            .astype(str)
            == str(
                selected_id
            )
        ]
        .iloc[0]
    )

    detail_1, detail_2, detail_3, detail_4 = (
        st.columns(4)
    )

    detail_1.metric(
        "Risk Score",

        f"{employee['Probability_Percent']:.2f}%",
    )

    detail_2.metric(
        "Risk Level",

        str(
            employee[
                "Risk_Level"
            ]
        ),
    )

    detail_3.metric(
        "Predicted Attrition",

        str(
            employee[
                "Predicted_Attrition"
            ]
        ),
    )

    detail_4.metric(
        "Priority Rank",

        f"#{int(employee['Priority_Rank'])}",
    )

    profile_columns = [
        "JobRole",
        "JobLevel",
        "OverTime",
        "MonthlyIncome",
        "YearsAtCompany",
        "StockOptionLevel",
        "WorkLifeBalance",
        "JobSatisfaction",
    ]

    profile = [
        {
            "Employee Attribute":
                column,

            "Value":
                employee[
                    column
                ],
        }

        for column
        in profile_columns

        if column
        in employee.index
    ]

    st.dataframe(
        pd.DataFrame(
            profile
        ),

        use_container_width=True,

        hide_index=True,
    )

    st.info(
        "Recommended action: "
        + str(
            employee[
                "Recommended_Action"
            ]
        )
    )


# ============================================================
# CSS
# ============================================================

st.html(
    """
    <style>

        :root {
    --navy:#17324d;
    --navy2:#234765;

    --blue:#647fbc;
    --blue-soft:#e8edf8;

    --red:#ef6b66;
    --red-soft:#fde9e7;

    --orange:#f4ad4e;
    --orange-soft:#fff1dd;

    --green:#62b99f;
    --green-soft:#e3f3ee;

    --text:#18304b;
    --muted:#56697f;

    --border:#e1e8f0;
    --background:#f7f9fc;
    --surface:#ffffff;
}

        .stApp {
            background:
                var(--background);

            color:
                var(--text);
        }

        .block-container {
            max-width:
                1600px;

            padding:
                1.2rem
                1.5rem
                3rem;
        }

        section[data-testid="stSidebar"] {

            background:
                linear-gradient(
                    180deg,
                    var(--navy),
                    var(--navy2)
                );

            border-right:
                none;
        }

        section[data-testid="stSidebar"] * {

            color:
                #ffffff;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"]
        label {

            border-radius:
                9px;

            padding:
                0.62rem
                0.72rem;

            margin-bottom:
                0.22rem;

            transition:
                0.15s;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"]
        label:hover {

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.10
                );
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked) {

            background:
                linear-gradient(
                    135deg,
                    #607bae,
                    #728ac2
                );

            box-shadow:
                0
                8px
                18px
                rgba(
                    0,
                    0,
                    0,
                    0.16
                );
        }

        .page-title {

            font-size:
                2rem;

            font-weight:
                800;

            line-height:
                1.08;

            color:
                var(--text);
        }

        .page-subtitle {

            color:
                var(--muted);

            margin:
                0.35rem
                0
                1rem;
        }

        .metric-card {

            background:
                #ffffff;

            border:
                1px solid
                var(--border);

            border-radius:
                14px;

            padding:
                1rem
                1.05rem;

            min-height:
                124px;

            position:
                relative;

            box-shadow:
                0
                6px
                20px
                rgba(
                    17,
                    38,
                    74,
                    0.04
                );
        }

.metric-card {

    background:
        var(--surface);

    border:
        1px solid
        var(--border);

    border-radius:
        18px;

    padding:
        1.05rem;

    min-height:
        132px;

    display:
        flex;

    align-items:
        flex-start;

    gap:
        0.9rem;

    box-shadow:
        0
        8px
        24px
        rgba(
            35,
            63,
            92,
            0.055
        );

    transition:
        transform
        0.18s ease,
        box-shadow
        0.18s ease;
}


.metric-card:hover {

    transform:
        translateY(
            -2px
        );

    box-shadow:
        0
        12px
        30px
        rgba(
            35,
            63,
            92,
            0.09
        );
}


.metric-icon {

    width:
        48px;

    height:
        48px;

    min-width:
        48px;

    border-radius:
        14px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    font-size:
        1.35rem;

    font-weight:
        800;
}


.icon-blue {

    color:
        #526ca7;

    background:
        var(--blue-soft);
}


.icon-red {

    color:
        #df5f5a;

    background:
        var(--red-soft);
}


.icon-orange {

    color:
        #d99234;

    background:
        var(--orange-soft);
}


.icon-green {

    color:
        #459d83;

    background:
        var(--green-soft);
}


.metric-content {

    flex:
        1;

    min-width:
        0;
}


.metric-label {

    color:
        #3f5269;

    font-size:
        0.82rem;

    font-weight:
        750;

    margin-top:
        0.05rem;
}


.metric-value {

    color:
        var(--text);

    font-size:
        1.9rem;

    font-weight:
        850;

    line-height:
        1;

    margin-top:
        0.72rem;
}


.metric-note {

    color:
        var(--muted);

    font-size:
        0.73rem;

    margin-top:
        0.72rem;

    line-height:
        1.35;
}
        .sidebar-brand {

            display:
                flex;

            align-items:
                center;

            gap:
                0.75rem;

            margin-bottom:
                1.35rem;
        }

        .sidebar-icon {

            width:
                44px;

            height:
                44px;

            border-radius:
                50%;

            border:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.28
                );

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            font-size:
                1.45rem;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.05
                );
        }

        .sidebar-status {

            border:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.22
                );

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.06
                );

            border-radius:
                12px;

            padding:
                0.85rem;

            margin-top:
                1.2rem;

            font-size:
                0.78rem;

            line-height:
                1.55;
        }

        .signal-row {

            display:
                flex;

            align-items:
                center;

            gap:
                0.7rem;

            border-bottom:
                1px solid
                #edf1f6;

            padding:
                0.62rem
                0;
        }

        .signal-row:last-child {

            border-bottom:
                none;
        }

        .signal-icon {

            width:
                34px;

            height:
                34px;

            min-width:
                34px;

            border-radius:
                9px;

            background:
                #eef4fb;

            color:
                #285f9e;

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            font-weight:
                800;
        }

        .signal-main {

            flex:
                1;
        }

        .signal-main small,
        .signal-high small,
        .signal-medium small,
        .signal-low small {

            display:
                block;

            color:
                #586b80;

            font-size:
                0.68rem;
        }

        .signal-high {

            color:
                #c7433f;

            text-align:
                right;
        }

        .signal-medium {

            color:
                #a96910;

            text-align:
                right;
        }

        .signal-low {

            color:
                #2f806a;

            text-align:
                right;
        }

        .intervention {

            background:
                #ffffff;

            border:
                1px solid
                var(--border);

            border-radius:
                14px;

            padding:
                1.1rem;

            min-height:
                235px;

            box-shadow:
                0
                6px
                20px
                rgba(
                    17,
                    38,
                    74,
                    0.035
                );
        }

        .pill {

            display:
                inline-block;

            border-radius:
                999px;

            padding:
                0.28rem
                0.58rem;

            font-size:
                0.73rem;

            font-weight:
                800;
        }

        .pill-high {

            color:
                #b72235;

            background:
                #fdecef;
        }

        .pill-medium {

            color:
                #9b6400;

            background:
                #fff4d8;
        }

        .pill-low {

            color:
                #1c7943;

            background:
                #e8f7ee;
        }

        div[data-testid="stDataFrame"] {

            border:
                1px solid
                var(--border);

            border-radius:
                12px;

            overflow:
                hidden;
        }


        /* ==================================================
           CLEAR CURRENT DATA BUTTON
           ================================================== */

        section[data-testid="stSidebar"]
        div.stButton,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] {

            margin-top:
                0.65rem;

            margin-bottom:
                0.20rem;
        }


        section[data-testid="stSidebar"]
        div.stButton > button,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button,

        section[data-testid="stSidebar"]
        button[kind="primary"] {

            width:
                100% !important;

            min-height:
                42px !important;

            background:
                linear-gradient(
                    135deg,
                    #eb706d,
                    #f08a78
                ) !important;

            color:
                #ffffff !important;

            border:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.30
                ) !important;

            border-radius:
                10px !important;

            box-shadow:
                0
                7px
                18px
                rgba(
                    0,
                    0,
                    0,
                    0.16
                ) !important;

            font-size:
                0.78rem !important;

            font-weight:
                750 !important;

            transition:
                0.16s !important;
        }


        section[data-testid="stSidebar"]
        div.stButton > button *,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button *,

        section[data-testid="stSidebar"]
        button[kind="primary"] * {

            color:
                #ffffff !important;

            fill:
                #ffffff !important;
        }


        section[data-testid="stSidebar"]
        div.stButton > button:hover,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button:hover,

        section[data-testid="stSidebar"]
        button[kind="primary"]:hover {

            background:
                linear-gradient(
                    135deg,
                    #e23650,
                    #ff5a7b
                ) !important;

            color:
                #ffffff !important;

            border-color:
                rgba(
                    255,
                    255,
                    255,
                    0.55
                ) !important;

            transform:
                translateY(
                    -1px
                );

            box-shadow:
                0
                10px
                22px
                rgba(
                    0,
                    0,
                    0,
                    0.22
                ) !important;
        }


        section[data-testid="stSidebar"]
        div.stButton > button:focus,

        section[data-testid="stSidebar"]
        div.stButton > button:active,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button:focus,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button:active {

            background:
                linear-gradient(
                    135deg,
                    #bd233a,
                    #e63e63
                ) !important;

            color:
                #ffffff !important;

            border-color:
                #ff9aac !important;

            outline:
                none !important;

            box-shadow:
                0
                0
                0
                3px
                rgba(
                    239,
                    71,
                    111,
                    0.25
                ) !important;
        }


        section[data-testid="stSidebar"]
        div.stButton > button p,

        section[data-testid="stSidebar"]
        div.stButton > button span,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button p,

        section[data-testid="stSidebar"]
        div[data-testid="stButton"] > button span {

            color:
                #ffffff !important;

            margin:
                0 !important;
        }


        /* ==================================================
           TEXT CONTRAST AND ACTION CONTROLS
           ================================================== */

        div[data-testid="stAlert"] {

            border:
                1px solid
                #c7d3df;

            box-shadow:
                0
                3px
                12px
                rgba(
                    35,
                    63,
                    92,
                    0.04
                );
        }

        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] li,
        div[data-testid="stAlert"] span {

            color:
                #263b53 !important;

            font-weight:
                600;
        }

        div[data-testid="stMetric"] {

            background:
                #ffffff;

            border:
                1px solid
                var(--border);

            border-radius:
                12px;

            padding:
                0.85rem
                0.95rem;
        }

        div[data-testid="stMetricLabel"] p,
        label[data-testid="stMetricLabel"] p {

            color:
                #40546b !important;

            font-weight:
                750 !important;
        }

        div[data-testid="stMetricValue"] {

            color:
                #18304b !important;

            font-weight:
                800 !important;
        }

        div[data-testid="stMetricDelta"] {

            color:
                #40546b !important;
        }

        div[data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] {

            color:
                #52667c !important;
        }

        div[data-testid="stDownloadButton"] > button,
        div[data-testid="stFileUploader"] button {

            background:
                #ffffff !important;

            color:
                #18304b !important;

            border:
                1px solid
                #91a4b8 !important;

            font-weight:
                750 !important;

            box-shadow:
                0
                3px
                10px
                rgba(
                    35,
                    63,
                    92,
                    0.05
                ) !important;
        }

        div[data-testid="stDownloadButton"] > button *,
        div[data-testid="stFileUploader"] button * {

            color:
                #18304b !important;

            fill:
                #18304b !important;
        }

        div[data-testid="stDownloadButton"] > button:hover,
        div[data-testid="stFileUploader"] button:hover {

            background:
                #edf3f8 !important;

            color:
                #102a43 !important;

            border-color:
                #58718a !important;
        }

        div[data-testid="stFileUploaderDropzone"] {

            background:
                #ffffff;

            border-color:
                #a6b6c6;
        }

        div[data-testid="stFileUploaderDropzone"] p,
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] span {

            color:
                #40546b !important;
        }

        main div[data-testid="stMarkdownContainer"] p,
        main div[data-testid="stMarkdownContainer"] li {

            color:
                #30465e;
        }

        .intervention h3 {

            color:
                #18304b;
        }

        .intervention li {

            color:
                #30465e;

            line-height:
                1.55;
        }

        header[data-testid="stHeader"] {

            background:
                transparent;
        }

        /* Keep Streamlit's sidebar collapse/reopen control available.
           Hiding the entire toolbar can also hide the button used to
           reopen a collapsed sidebar. */
        div[data-testid="stToolbar"] {

            visibility:
                visible !important;

            height:
                auto !important;
        }

        div[data-testid="stToolbarActions"] {

            visibility:
                hidden !important;
        }

        button[data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {

            visibility:
                visible !important;

            opacity:
                1 !important;

            pointer-events:
                auto !important;

            z-index:
                1000000 !important;
        }

        button[data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapsedControl"] button {

            position:
                fixed !important;

            top:
                0.75rem !important;

            left:
                0.75rem !important;

            width:
                42px !important;

            height:
                42px !important;

            min-width:
                42px !important;

            border-radius:
                10px !important;

            background:
                var(--navy) !important;

            color:
                #ffffff !important;

            border:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.28
                ) !important;

            box-shadow:
                0
                6px
                18px
                rgba(
                    17,
                    38,
                    74,
                    0.20
                ) !important;
        }

        button[data-testid="stSidebarCollapsedControl"] *,
        [data-testid="stSidebarCollapsedControl"] button * {

            color:
                #ffffff !important;

            fill:
                #ffffff !important;
        }

        #MainMenu,
        footer {

            visibility:
                hidden;
        }

    </style>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-icon">
                👥
            </div>

            <div>

                <b>
                    Employee Attrition
                </b>

                <br>

                <small>
                    Early Warning System
                </small>

            </div>

        </div>
        """
    )

    page = st.radio(
        "Navigation",

        [
            "🏠  Overview",
            "👥  Employee List",
            "⚠️  Risk Profiles",
            "🛠️  Interventions",
            "📤  Upload Data",
            "ℹ️  About",
        ],

        label_visibility=(
            "collapsed"
        ),
    )

    if (
        st.session_state[
            "results"
        ]
        is not None
    ):

        st.html(
            f"""
            <div class="sidebar-status">

                <b>
                    CURRENT DATA
                </b>

                <br>

                {
                    st.session_state[
                        "filename"
                    ]
                }

                <br>

                {
                    st.session_state[
                        "processed_at"
                    ]
                    or "Current session"
                }

            </div>
            """
        )

        if st.button(
            "🗑️ Clear current data",

            type="primary",

            use_container_width=True,

            key=(
                "clear_current_data_button"
            ),
        ):

            for (
                key,
                default_value,
            ) in DEFAULT_STATE.items():

                st.session_state[
                    key
                ] = default_value

            st.session_state.pop(
                "employee_file_uploader",
                None,
            )

            st.rerun()

    st.html(
        f"""
        <div class="sidebar-status">

            <b>
                MODEL STATUS
            </b>

            <br>

            ● Model Ready

            <br>

            {model_name}

            <br>

            Threshold
            {prediction_threshold:.2f}

        </div>
        """
    )

    st.caption(
        "Developed by PBHI Solutions"
    )


results = (
    st.session_state[
        "results"
    ]
)

uploaded_data = (
    st.session_state[
        "uploaded_data"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠  Overview":

    st.html(
        """
        <div class="page-title">
            Employee Attrition
            Early Warning System
        </div>

        <div class="page-subtitle">
            Monitor risk,
            understand employee profiles,
            and prioritize appropriate follow-up.
        </div>
        """
    )

    if results is None:

        st.info(
            "Upload data karyawan untuk "
            "membuka dashboard interaktif."
        )

        left, right = (
            st.columns(
                [
                    1.4,
                    1,
                ],

                gap="large",
            )
        )

        with left:

            st.subheader(
                "Upload Employee Data"
            )

            upload_component()

        with right:

            st.subheader(
                "Dashboard Output"
            )

            st.markdown(
                """
                - distribusi Low, Medium,
                  dan High Risk;
                - High Risk Rate berdasarkan
                  Job Role;
                - average risk berdasarkan
                  Job Level;
                - Top 10 Priority Employees;
                - Employee Detail View;
                - Monitoring Action Queue.
                """
            )

    else:

        job_levels = [
            "All",

            *[
                str(
                    value
                )

                for value in sorted(

                    pd.to_numeric(
                        results[
                            "JobLevel"
                        ],

                        errors="coerce",
                    )
                    .dropna()
                    .unique()
                )
            ],
        ]

        job_roles = [
            "All",

            *sorted(
                results[
                    "JobRole"
                ]
                .dropna()
                .astype(str)
                .unique()
            ),
        ]

        overtime_values = [
            "All",

            *sorted(
                results[
                    "OverTime"
                ]
                .dropna()
                .astype(str)
                .unique()
            ),
        ]

        with st.container(
            border=True
        ):

            f1, f2, f3, f4 = (
                st.columns(
                    [
                        1,
                        1.35,
                        1,
                        1,
                    ],

                    gap="large",
                )
            )

            with f1:

                job_level_filter = (
                    st.selectbox(
                        "Job Level",

                        job_levels,

                        key=(
                            "overview_level"
                        ),
                    )
                )

            with f2:

                job_role_filter = (
                    st.selectbox(
                        "Job Role",

                        job_roles,

                        key=(
                            "overview_role"
                        ),
                    )
                )

            with f3:

                overtime_filter = (
                    st.selectbox(
                        "OverTime",

                        overtime_values,

                        key=(
                            "overview_overtime"
                        ),
                    )
                )

            with f4:

                st.markdown(
                    "**Data processed**"
                )

                st.caption(
                    st.session_state[
                        "processed_at"
                    ]
                    or "Current session"
                )

        filtered = (
            filter_results(
                results,

                job_level_filter,

                job_role_filter,

                overtime_filter,
            )
        )

        if filtered.empty:

            st.warning(
                "Tidak ada karyawan yang "
                "sesuai dengan filter."
            )

            st.stop()

        summary = (
            get_summary(
                filtered
            )
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            metric_card(
                "Total Employees",

                f"{summary['total']:,}",

                "Employees in current selection",

                "icon-blue",

                "👥",
            )

        with c2:

            metric_card(
                "High Risk Employees",

                f"{summary['high']:,}",

                (
                    f"{summary['high'] / summary['total']:.1%} "
                    "of selected employees"
                ),

                "icon-red",

                "⚠️",
            )

        with c3:

            metric_card(
                "Average Risk Score",

                f"{summary['average']:.1%}",

                "Mean predicted attrition probability",

                "icon-orange",

                "📊",
            )

        with c4:

            metric_card(
                "Monitoring Queue",

                f"{summary['monitoring']:,}",

                (
                    f"{summary['monitoring'] / summary['total']:.1%} "
                    "Medium + High Risk"
                ),

                "icon-green",

                "📄",
            )

        st.write("")

        chart_1, chart_2, chart_3 = (
            st.columns(
                [
                    0.95,
                    1.65,
                    1.05,
                ],

                gap="large",
            )
        )

        with chart_1:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Risk Distribution"
                )

                pie_data = (
                    risk_distribution(
                        filtered
                    )
                )

                figure = px.pie(
                    pie_data,

                    names=(
                        "Risk Level"
                    ),

                    values=(
                        "Employees"
                    ),

                    hole=0.60,

                    color=(
                        "Risk Level"
                    ),

                    color_discrete_map={
                        "Low Risk":
                            "#62b99f",

                        "Medium Risk":
                            "#f4ad4e",

                        "High Risk":
                            "#ef6b66",
                    },
                )

                figure.update_traces(
                    textposition="inside",

                    textinfo="percent",

                    textfont=dict(
                        color="#ffffff",
                        size=13,
                    ),

                    marker=dict(
                        line=dict(
                            color='#ffffff',
                            width=2,
                        )
                    ),
                )

                figure.add_annotation(
                    text=(
                        f"<b>{summary['total']}</b>"
                        "<br>Total"
                    ),

                    x=0.5,

                    y=0.5,

                    showarrow=False,
                )

                clean_plot(
                    figure,
                    330,
                )

                figure.update_layout(
                    legend=dict(
                        orientation="h",

                        y=-0.12,

                        x=0.5,

                        xanchor="center",
                    )
                )

                st.plotly_chart(
                    figure,

                    use_container_width=True,

                    config={
                        "displayModeBar":
                            False,
                    },
                )
        with chart_2:
            with st.container(border=True):
                st.subheader("Risk by Job Role")
                st.caption("Average attrition risk score for each employee role.")

                role_data = (
                    job_role_summary(filtered)
                    .sort_values("Average Risk Score", ascending=True)
                    .tail(7)
                    .copy()
                )

                soft_role_colors = [
                    "#73b9aa",
                    "#86bda0",
                    "#a7c47a",
                    "#d0c25f",
                    "#edb55d",
                    "#ef9367",
                    "#eb6f6b",
                ]

                figure = go.Figure()

                figure.add_trace(
                    go.Bar(
                        x=role_data["Average Risk Score"],
                        y=role_data["JobRole"],
                        orientation="h",
                        marker=dict(
                            color=soft_role_colors[-len(role_data):],
                            line=dict(width=0),
                        ),
                        customdata=np.stack(
                            [
                                role_data["Employees"],
                                role_data["High_Risk_Employees"],
                                role_data["High Risk Rate"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Employees: %{customdata[0]:,.0f}<br>"
                            "High Risk Employees: %{customdata[1]:,.0f}<br>"
                            "High Risk Rate: %{customdata[2]:.1f}%<br>"
                            "Average Risk Score: %{x:.1f}%"
                            "<extra></extra>"
                        ),
                    )
                )

                max_role_risk = float(
                    role_data[
                        "Average Risk Score"
                    ].max()
                )

                role_axis_max = max(
                    max_role_risk,
                    1.0,
                )

                role_label_offset = (
                    role_axis_max
                    * 0.025
                )

                role_label_positions = (
                    role_data[
                        "Average Risk Score"
                    ]
                    .clip(
                        lower=(
                            role_label_offset
                        )
                    )
                )

                figure.add_trace(
                    go.Scatter(
                        x=role_label_positions,
                        y=role_data["JobRole"],
                        mode="text",
                        text=[
                            f"{value:.1f}%"
                            for value in role_data[
                                "Average Risk Score"
                            ]
                        ],
                        textposition="middle right",
                        textfont=dict(
                            color="#263b53",
                            size=12,
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                        cliponaxis=False,
                    )
                )

                figure.update_layout(
                    height=360,
                    margin=dict(l=10, r=55, t=10, b=35),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    font=dict(family="Arial", color="#263b53", size=12),
                    xaxis=dict(
                        title="Average Risk Score",
                        ticksuffix="%",
                        range=[0, role_axis_max * 1.25],
                        showgrid=True,
                        gridcolor="#e3e9f0",
                        zeroline=False,
                        fixedrange=True,
                        tickfont=dict(color="#344a62"),
                        title_font=dict(color="#263b53"),
                    ),
                    yaxis=dict(
                        title="",
                        showgrid=False,
                        fixedrange=True,
                        tickfont=dict(size=11, color="#263b53"),
                    ),
                    bargap=0.42,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                st.caption(
                    "Employee counts and High Risk Rate are available "
                    "when hovering over each bar."
                )

        with chart_3:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Risk by Job Level"
                )

                level_data = (
                    job_level_summary(
                        filtered
                    )
                )

                figure = px.bar(
                    level_data,

                    x="JobLevel",

                    y=(
                        "Average Risk Score"
                    ),

                    text=(
                        "Average Risk Score"
                    ),

                    color=(
                        "Average Risk Score"
                    ),

                    color_continuous_scale=[
                        "#62b99f",
                        "#a7c47a",
                        "#f4ad4e",
                        "#ef6b66",
                    ],

                    hover_data={
                        "Employees":
                            True,
                    },
                )

                figure.update_traces(
                    texttemplate=(
                        "%{text:.1f}%"
                    ),

                    textposition=(
                        "outside"
                    ),

                    textfont=dict(
                        color="#263b53",
                        size=12,
                    ),

                    cliponaxis=False,
                )

                figure.update_layout(
                    coloraxis_showscale=False,

                    showlegend=False,

                    xaxis_title=(
                        "Job Level"
                    ),

                    yaxis_title=(
                        "Average Risk Score"
                    ),

                    xaxis=dict(
                        tickmode="array",

                        tickvals=(
                            level_data[
                                "JobLevel"
                            ]
                            .tolist()
                        ),

                        ticktext=[
                            str(value)
                            for value
                            in level_data[
                                "JobLevel"
                            ]
                            .tolist()
                        ],

                        tickfont=dict(
                            color="#263b53",
                            size=12,
                        ),

                        title_font=dict(
                            color="#263b53",
                            size=14,
                        ),

                        linecolor="#8ea0b2",

                        gridcolor="#e3e9f0",

                        zeroline=False,
                    ),

                    yaxis=dict(
                        ticksuffix="%",

                        rangemode="tozero",

                        tickfont=dict(
                            color="#263b53",
                            size=12,
                        ),

                        title_font=dict(
                            color="#263b53",
                            size=14,
                        ),

                        gridcolor="#d8e0e8",

                        linecolor="#8ea0b2",

                        zeroline=False,
                    ),

                    legend=dict(
                        font=dict(
                            color="#263b53",
                            size=12,
                        )
                    ),
                )

                clean_plot(
                    figure,
                    330,
                )

                st.plotly_chart(
                    figure,

                    use_container_width=True,

                    config={
                        "displayModeBar":
                            False,
                    },
                )

        with st.container(
            border=True
        ):

            st.subheader(
                "Top 10 Priority Employees"
            )

            prediction_table(
                filtered,

                limit=10,
            )

        st.caption(
            f"Low Risk < "
            f"{low_medium_cutoff:.0%}; "

            f"Medium Risk "
            f"{low_medium_cutoff:.0%}–"
            f"<{medium_high_cutoff:.0%}; "

            f"High Risk ≥ "
            f"{medium_high_cutoff:.0%}. "

            "Predictions support HR review "
            "and are not proof that an "
            "employee will resign."
        )


# ============================================================
# EMPLOYEE LIST
# ============================================================

elif page == "👥  Employee List":

    st.html(
        """
        <div class="page-title">
            Employee List
        </div>

        <div class="page-subtitle">
            Search, filter, and review
            employee-level predictions.
        </div>
        """
    )

    if results is None:

        st.info(
            "Upload data terlebih dahulu."
        )

    else:

        f1, f2, f3 = (
            st.columns(
                [
                    1.2,
                    1.2,
                    1.3,
                ]
            )
        )

        with f1:

            risk_filter = (
                st.multiselect(
                    "Risk Level",

                    [
                        "High Risk",
                        "Medium Risk",
                        "Low Risk",
                    ],

                    default=[
                        "High Risk",
                        "Medium Risk",
                        "Low Risk",
                    ],
                )
            )

        with f2:

            role_filter = (
                st.selectbox(
                    "Job Role",

                    [
                        "All",

                        *sorted(
                            results[
                                "JobRole"
                            ]
                            .astype(str)
                            .unique()
                        ),
                    ],

                    key=(
                        "employee_role"
                    ),
                )
            )

        with f3:

            employee_search = (
                st.text_input(
                    "Search Employee ID",

                    placeholder=(
                        "Example: 10004"
                    ),
                )
            )

        filtered = (
            results[
                results[
                    "Risk_Level"
                ]
                .isin(
                    risk_filter
                )
            ]
            .copy()
        )

        if role_filter != "All":

            filtered = filtered[
                filtered[
                    "JobRole"
                ]
                == role_filter
            ]

        if employee_search.strip():

            filtered = filtered[
                filtered[
                    "EmployeeNumber"
                ]
                .astype(str)
                .str.contains(
                    employee_search
                    .strip(),

                    case=False,

                    na=False,
                )
            ]

        st.write(
            f"Menampilkan "
            f"{len(filtered):,} "
            f"karyawan."
        )

        prediction_table(
            filtered
        )

        download_button(
            filtered,

            "PBHI_filtered_employee_results.csv",
        )

        st.divider()

        st.subheader(
            "Employee Detail View"
        )

        if filtered.empty:

            st.info(
                "Tidak ada employee untuk ditampilkan."
            )

        else:

            employee_detail(
                filtered
            )


# ============================================================
# RISK PROFILES
# ============================================================

elif page == "⚠️  Risk Profiles":

    st.html(
        """
        <div class="page-title">
            Risk Profiles
        </div>

        <div class="page-subtitle">
            Descriptive profiles associated
            with higher model risk.
        </div>
        """
    )

    if results is None:

        st.info(
            "Upload data terlebih dahulu."
        )

    else:

        st.warning(
            "Bagian ini menunjukkan pola "
            "deskriptif, bukan bukti "
            "sebab-akibat dan bukan "
            "individual SHAP."
        )

        left, right = (
            st.columns(
                [
                    1,
                    1.25,
                ],

                gap="large",
            )
        )

        with left:

            with st.container(
                border=True
            ):

                st.subheader(
                    "High-Risk Profile Signals"
                )

                signal_panel(
                    results
                )

        with right:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Average Risk by OverTime"
                )

                overtime_data = (
                    results
                    .groupby(
                        "OverTime"
                    )
                    .agg(

                        Employees=(
                            "EmployeeNumber",
                            "size",
                        ),

                        Average_Risk=(
                            "Attrition_Probability",
                            "mean",
                        ),
                    )
                    .reset_index()
                )

                overtime_data[
                    "Average Risk Score"
                ] = (
                    overtime_data[
                        "Average_Risk"
                    ]
                    * 100
                )

                figure = px.bar(
                    overtime_data,

                    x="OverTime",

                    y=(
                        "Average Risk Score"
                    ),

                    text=(
                        "Average Risk Score"
                    ),

                    color="OverTime",

                    color_discrete_map={
                        "Yes":
                            "#ef3d4f",

                        "No":
                            "#2dae61",
                    },

                    hover_data={
                        "Employees":
                            True,
                    },
                )

                figure.update_traces(
                    texttemplate=(
                        "%{text:.1f}%"
                    ),

                    textposition=(
                        "outside"
                    ),

                    textfont=dict(
                        color="#263b53",
                        size=12,
                    ),

                    cliponaxis=False,
                )

                figure.update_layout(
                    showlegend=False,

                    xaxis=dict(
                        title=dict(
                            text="OverTime",
                            font=dict(
                                color="#263b53",
                                size=14,
                            ),
                        ),

                        tickfont=dict(
                            color="#263b53",
                            size=12,
                        ),

                        linecolor="#8ea0b2",

                        gridcolor="#e3e9f0",

                        zeroline=False,
                    ),

                    yaxis=dict(
                        title=dict(
                            text="Average Risk Score",
                            font=dict(
                                color="#263b53",
                                size=14,
                            ),
                        ),

                        ticksuffix="%",

                        rangemode="tozero",

                        tickfont=dict(
                            color="#263b53",
                            size=12,
                        ),

                        linecolor="#8ea0b2",

                        gridcolor="#d8e0e8",

                        zeroline=False,
                    ),
                )

                clean_plot(
                    figure,
                    360,
                )

                st.plotly_chart(
                    figure,

                    use_container_width=True,

                    config={
                        "displayModeBar":
                            False,
                    },
                )


# ============================================================
# INTERVENTIONS
# ============================================================

elif page == "🛠️  Interventions":

    st.html(
        """
        <div class="page-title">
            Interventions
        </div>

        <div class="page-subtitle">
            Suggested review priorities
            based on risk segmentation.
        </div>
        """
    )

    c1, c2, c3 = (
        st.columns(
            3,

            gap="large",
        )
    )

    with c1:

        st.html(
            """
            <div class="intervention">

                <span class="pill pill-high">
                    HIGH RISK
                </span>

                <h3>
                    Immediate HR Review
                </h3>

                <ul>

                    <li>
                        Schedule a confidential check-in.
                    </li>

                    <li>
                        Review workload and overtime.
                    </li>

                    <li>
                        Discuss career and compensation concerns.
                    </li>

                    <li>
                        Document agreed follow-up actions.
                    </li>

                </ul>

            </div>
            """
        )

    with c2:

        st.html(
            """
            <div class="intervention">

                <span class="pill pill-medium">
                    MEDIUM RISK
                </span>

                <h3>
                    Monitor and Follow Up
                </h3>

                <ul>

                    <li>
                        Review in the next HR cycle.
                    </li>

                    <li>
                        Monitor workload and tenure milestones.
                    </li>

                    <li>
                        Confirm development opportunities.
                    </li>

                    <li>
                        Escalate only when context supports it.
                    </li>

                </ul>

            </div>
            """
        )

    with c3:

        st.html(
            """
            <div class="intervention">

                <span class="pill pill-low">
                    LOW RISK
                </span>

                <h3>
                    Standard Engagement
                </h3>

                <ul>

                    <li>
                        Maintain normal check-ins.
                    </li>

                    <li>
                        Continue recognition and development.
                    </li>

                    <li>
                        Re-score when new data is available.
                    </li>

                    <li>
                        Low Risk does not mean zero risk.
                    </li>

                </ul>

            </div>
            """
        )

    if results is not None:

        st.divider()

        st.subheader(
            "Monitoring Action Queue"
        )

        queue = (
            results[
                results[
                    "Risk_Level"
                ]
                .isin(
                    [
                        "High Risk",
                        "Medium Risk",
                    ]
                )
            ]
            .copy()
        )

        if queue.empty:

            st.success(
                "Tidak ada employee "
                "dalam monitoring queue."
            )

        else:

            queue[
                "HR Status"
            ] = (
                queue[
                    "EmployeeNumber"
                ]
                .astype(str)
                .map(
                    st.session_state[
                        "intervention_status"
                    ]
                )
                .fillna(
                    "Not Reviewed"
                )
            )

            queue_view = (
                queue[
                    [
                        "EmployeeNumber",
                        "Probability_Percent",
                        "Risk_Level",
                        "Recommended_Action",
                        "HR Status",
                    ]
                ]
                .rename(
                    columns={

                        "EmployeeNumber":
                            "Employee ID",

                        "Probability_Percent":
                            "Risk Score",

                        "Risk_Level":
                            "Risk Level",

                        "Recommended_Action":
                            "Recommended Action",
                    }
                )
            )

            edited = (
                st.data_editor(
                    queue_view,

                    use_container_width=True,

                    hide_index=True,

                    disabled=[
                        "Employee ID",
                        "Risk Score",
                        "Risk Level",
                        "Recommended Action",
                    ],

                    column_config={

                        "Risk Score":
                            st.column_config.ProgressColumn(

                                "Risk Score",

                                min_value=0,

                                max_value=100,

                                format="%.2f%%",
                            ),

                        "HR Status":
                            st.column_config.SelectboxColumn(

                                "HR Status",

                                options=[
                                    "Not Reviewed",
                                    "Review Scheduled",
                                    "Follow-up in Progress",
                                    "Follow-up Completed",
                                ],

                                required=True,
                            ),
                    },
                )
            )

            for (
                _,
                row,
            ) in edited.iterrows():

                st.session_state[
                    "intervention_status"
                ][
                    str(
                        row[
                            "Employee ID"
                        ]
                    )
                ] = (
                    row[
                        "HR Status"
                    ]
                )

            download_button(
                edited,

                "PBHI_monitoring_action_queue.csv",

                "⬇️ Download Monitoring Queue",
            )


# ============================================================
# UPLOAD DATA
# ============================================================

elif page == "📤  Upload Data":

    st.html(
        """
        <div class="page-title">
            Upload Data
        </div>

        <div class="page-subtitle">
            Upload employee data and
            generate attrition-risk predictions.
        </div>
        """
    )

    left, right = (
        st.columns(
            [
                1.45,
                1,
            ],

            gap="large",
        )
    )

    with left:

        upload_component()

    with right:

        st.subheader(
            "Input Requirements"
        )

        st.markdown(
            f"""
            - **{len(required_columns)}**
              required raw columns;
            - **{len(selected_features)}**
              final model features;
            - two engineered features are
              created automatically;
            - SMOTE is applied during
              training only;
            - `EmployeeNumber` is used
              only as Employee ID.
            """
        )

        st.info(
            "Gunakan data demo atau data "
            "yang sudah dianonimkan."
        )

    if (
        st.session_state[
            "quality"
        ]
        is not None
    ):

        st.divider()

        st.subheader(
            "Data Quality Summary"
        )

        quality = (
            st.session_state[
                "quality"
            ]
        )

        q1, q2, q3, q4 = (
            st.columns(4)
        )

        q1.metric(
            "Rows Loaded",

            f"{quality['rows']:,}",
        )

        q2.metric(
            "Required Columns",

            (
                f"{quality['required_available']}"
                f"/"
                f"{quality['required_total']}"
            ),
        )

        q3.metric(
            "Missing Cells",

            f"{quality['missing_cells']:,}",
        )

        q4.metric(
            "Duplicate Employee IDs",

            f"{quality['duplicate_ids']:,}",
        )

        if quality[
            "missing_required"
        ]:

            st.error(
                "Kolom wajib yang belum tersedia: "
                + ", ".join(
                    quality[
                        "missing_required"
                    ]
                )
            )

        else:

            st.success(
                "Struktur kolom wajib sudah lengkap."
            )

    if uploaded_data is not None:

        st.subheader(
            "Current Uploaded Data"
        )

        st.dataframe(
            uploaded_data
            .head(20),

            use_container_width=True,

            hide_index=True,
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️  About":

    st.html(
        """
        <div class="page-title">
            About
        </div>

        <div class="page-subtitle">
            PBHI Employee Attrition
            Early Warning System
        </div>
        """
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    m1.metric(
        "Test F2 Score",
        "65.52%",
    )

    m2.metric(
        "Test Recall",
        "80.85%",
    )

    m3.metric(
        "Prediction Threshold",

        f"{prediction_threshold:.2f}",
    )

    m4.metric(
        "Final Features",

        str(
            len(
                selected_features
            )
        ),
    )

    st.markdown(
        f"""
        ### Model Configuration

        - **Model:** {model_name}
        - **Sampling:** SMOTE 0.6 
        - **Low Risk:** probability < {low_medium_cutoff:.0%}
        - **Medium Risk:** {low_medium_cutoff:.0%}
          ≤ probability < {medium_high_cutoff:.0%}
        - **High Risk:** probability ≥ {medium_high_cutoff:.0%}
        - **Prediction purpose:** Early warning and HR decision support tool
        """
    )

    st.warning(
        "Predictions are decision-support information, "
        "not proof that an employee will resign. "
        "The result must not be used as the sole basis "
        "for termination, promotion, compensation, "
        "or disciplinary decisions."
    )
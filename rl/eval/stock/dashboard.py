import streamlit as st
from mlflow_utils import artifact_path, init_mlflow, load_file
from mlflow_utils import setup_logger
import utils.paths
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

EXPERIMENT = "Tests"
POINT_DATA_FN = "eval_stockwise.json"
METADATA_FN = "agg_metadata.csv"
AGG_FN = "agg_metadata_stats.csv"


@st.cache
def plot_price_action(ticker_data):
    df = pd.DataFrame(ticker_data["points"])
    df["actions"] = df["actions"].astype(str)

    fig1 = px.scatter(x="timesteps", y="prices", data_frame=df, color="actions", hover_data=["rewards"])
    fig1.update_traces(marker={"size": 20})

    fig2 = px.line(x="timesteps", y="prices", data_frame=df)

    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_layout(
        title="",
        xaxis_title="Timesteps",
        yaxis_title="Price (scaled)",
        legend_title="Actions",
    )
    fig3.update(layout_yaxis_range=[0, 1])

    return fig3


st.set_page_config(layout="wide")

mlflow_id = st.sidebar.text_input(label="mlflow id", value="")

if mlflow_id != "":
    init_mlflow(utils.paths.mlflow_dir, "Tests")
    setup_logger("INFO")

    point_data = load_file(run_id=mlflow_id, fn=POINT_DATA_FN, experiment=EXPERIMENT)
    metadata = load_file(run_id=mlflow_id, fn=METADATA_FN, experiment=EXPERIMENT)
    agg = load_file(run_id=mlflow_id, fn=AGG_FN, experiment=EXPERIMENT)

    with st.expander("Aggregated Statistics"):
        col1, col2 = st.columns(2)

        col1.dataframe(agg)
        col2.dataframe(metadata)

    ticker_names = [ticker["metadata"]["ticker"] for ticker in point_data]

    ticker_name = st.sidebar.selectbox(
        "Stock ticker", tuple(ticker_names),
        index=0
    )
    ticker_id = ticker_names.index(ticker_name)

    curr_ticker = point_data[ticker_id]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(label="Profit", value=round(curr_ticker["metadata"]["profit"], 3))
    col2.metric(label="N_Open_Positions", value=curr_ticker["metadata"]["open_positions"])
    col3.metric(label="Min Date", value=curr_ticker["metadata"]["min_date"])
    col4.metric(label="Max Date", value=curr_ticker["metadata"]["max_date"])

    # col2.metric(label="Open Positions", value=curr_ticker["metadata"]["open_positions"])

    st.plotly_chart(plot_price_action(curr_ticker), use_container_width=True)

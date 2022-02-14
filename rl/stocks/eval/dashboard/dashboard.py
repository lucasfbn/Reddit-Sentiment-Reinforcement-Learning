import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlflow_utils import init_mlflow, load_file
from mlflow_utils import setup_logger

import utils.paths

EXPERIMENT = "Eval_Stocks"
EVAL_TICKER_FN = "evl_ticker.pkl"
AGG_EVAL_DF_FN = "agg_evl_df.csv"
AGG_STATS_FN = "agg_stats.csv"


@st.cache
def plot_price_action(ticker):
    df = pd.DataFrame([seq.evl.to_dict() | seq.metadata.to_dict() for seq in ticker.sequences])
    df["date"] = df["date"].astype(str)
    df["action"] = df["action"].astype(str)

    fig1 = px.scatter(x=df.index, y="price", data_frame=df, color="action", hover_data=["reward", "date"],
                      color_discrete_map={'2': 'rgb(255,0,0)', '1': 'rgb(0,255,0)',
                                          '0': 'rgb(0,0,255)'})
    fig1.update_traces(marker={"size": 20})

    fig2 = px.line(x=df.index, y="price", data_frame=df)

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

    eval_ticker = load_file(run_id=mlflow_id, fn=EVAL_TICKER_FN, experiment=EXPERIMENT)
    agg_evl_df = load_file(run_id=mlflow_id, fn=AGG_EVAL_DF_FN, experiment=EXPERIMENT)
    agg_stats = load_file(run_id=mlflow_id, fn=AGG_STATS_FN, experiment=EXPERIMENT)

    with st.expander("Aggregated Statistics"):
        col1, col2 = st.columns(2)

        col1.dataframe(agg_stats)
        col2.dataframe(agg_evl_df)

    ticker_names = [ticker.name for ticker in eval_ticker]

    ticker_name = st.sidebar.selectbox(
        "Stock ticker", tuple(ticker_names),
        index=0
    )
    ticker_id = ticker_names.index(ticker_name)

    curr_ticker = eval_ticker[ticker_id]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(label="Profit", value=round(curr_ticker.evl.reward, 3))
    col2.metric(label="N_Open_Positions", value=curr_ticker.evl.open_positions)
    col3.metric(label="Min Date", value=str(curr_ticker.evl.min_date))
    col4.metric(label="Max Date", value=str(curr_ticker.evl.max_date))

    st.plotly_chart(plot_price_action(curr_ticker), use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Miller Construction RFP Analyzer", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🏗️ Miller Construction Company")
st.subheader("RFP Bidding Simulation & Optimization")
st.markdown("Analyze expected profit and find the optimal bid amount using Monte Carlo simulation under triangular and historical distributions.")

# --- SIDEBAR: SIMULATION INPUTS ---
st.sidebar.header("⚙️ Simulation Settings")
n_iterations = st.sidebar.slider("Number of Iterations", min_value=1000, max_value=50000, value=10000, step=1000)

st.sidebar.header("📋 Miller's Bid Parameters")
miller_bid = st.sidebar.number_input("Proposed Bid Amount ($)", min_value=0, value=175000, step=1000)
use_optimal_curve = st.sidebar.checkbox("Analyze Optimal Bid Curve", value=True, help="Simulates across a range of bids to find the optimal.")

if use_optimal_curve:
    bid_range_min = st.sidebar.number_input("Bid Scan Min ($)", value=120000, step=1000)
    bid_range_max = st.sidebar.number_input("Bid Scan Max ($)", value=220000, step=1000)
    bid_range_step = st.sidebar.number_input("Bid Scan Step ($)", value=5000, step=1000)

st.sidebar.header("🏗️ Project & Prep Costs")
st.sidebar.caption("⚠️ Constraint: Project costs cannot be < $70,000")
data_source = st.sidebar.radio("Data Source for Costs", ["Historical Data (CSV)", "Triangular Distribution", "Constant Fixed Cost"])

if data_source == "Triangular Distribution":
    st.sidebar.markdown("**Project Cost**")
    proj_min = st.sidebar.number_input("Proj Cost Min ($)", min_value=70000, value=80000, step=1000)
    proj_mode = st.sidebar.number_input("Proj Cost Mode ($)", min_value=70000, value=100000, step=1000)
    proj_max = st.sidebar.number_input("Proj Cost Max ($)", min_value=70000, value=150000, step=1000)
    
    st.sidebar.markdown("**Bid Preparation Cost**")
    prep_cost_val = st.sidebar.number_input("Bid Prep Cost ($)", value=1500, step=100)

elif data_source == "Constant Fixed Cost":
    st.sidebar.markdown("**Fixed Costs**")
    fixed_proj_cost = st.sidebar.number_input("Fixed Project Cost ($)", min_value=70000, value=70000, step=1000)
    prep_cost_val = st.sidebar.number_input("Bid Prep Cost ($)", value=1500, step=100)

else:
    st.sidebar.info("Upload a custom CSV or use the default `project_costs.csv`")
    uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])
    if not uploaded_file and not os.path.exists("project_costs.csv"):
        st.sidebar.error("No CSV found! Please upload one or switch to Triangular.")

st.sidebar.header("🤝 Competitor Analysis")
num_competitors = st.sidebar.number_input("Number of Competitors", min_value=1, max_value=10, value=3)

competitors = []
for i in range(int(num_competitors)):
    # Match default values from the Miller class case slide
    default_prob = 1.0 if i == 0 else 0.5
    with st.sidebar.expander(f"Competitor {chr(65+i)}", expanded=(i<3)):
        prob_bid = st.slider(f"Prob. of Bidding", 0.0, 1.0, default_prob, key=f"prob_{i}")
        c_min = st.number_input("Bid Min ($)", value=90000, step=1000, key=f"cmin_{i}")
        c_mode = st.number_input("Bid Mode ($)", value=130000, step=1000, key=f"cmode_{i}")
        c_max = st.number_input("Bid Max ($)", value=180000, step=1000, key=f"cmax_{i}")
        competitors.append({"prob": prob_bid, "min": c_min, "mode": c_mode, "max": c_max})


# --- SIMULATION LOGIC ---
@st.cache_data
def load_csv_data(file_content=None):
    if file_content is not None:
        df = pd.read_csv(file_content)
    elif os.path.exists("project_costs.csv"):
        df = pd.read_csv("project_costs.csv")
    else:
        return None, None
        
    try:
        if 'Total Project Costs (exluding bid preparation costs)' in df.columns:
            proj_col = 'Total Project Costs (exluding bid preparation costs)'
        elif 'Total Project Costs' in df.columns:
            proj_col = 'Total Project Costs'
        else:
            proj_col = df.columns[1]
            
        if 'Bid Preparation Costs' in df.columns:
            prep_col = 'Bid Preparation Costs'
        else:
            prep_col = df.columns[0]
            
        df_proj = df[proj_col].dropna().values
        df_prep = df[prep_col].dropna().values
        return df_proj, df_prep
    except Exception as e:
        return None, None

def run_simulation(bid_amount, iterations):
    np.random.seed(42) # Fixed seed for stable optimal curve
    
    if data_source == "Triangular Distribution":
        project_costs = np.random.triangular(proj_min, proj_mode, proj_max, iterations)
        # Note: Total project completion costs can't ever be less than $70,000 as per case constraints
        project_costs = np.maximum(project_costs, 70000)
        prep_costs = np.full(iterations, prep_cost_val)
    elif data_source == "Constant Fixed Cost":
        project_costs = np.full(iterations, fixed_proj_cost)
        project_costs = np.maximum(project_costs, 70000)
        prep_costs = np.full(iterations, prep_cost_val)
    else:
        file_to_load = uploaded_file if 'uploaded_file' in globals() or 'uploaded_file' in locals() else None
        df_proj, df_prep = load_csv_data(file_to_load)
        if df_proj is not None and len(df_proj) > 0:
            project_costs = np.random.choice(df_proj, iterations)
            # Ensure project costs are never less than 70,000 as per case constraints
            project_costs = np.maximum(project_costs, 70000)
            prep_costs = np.random.choice(df_prep, iterations)
        else:
            project_costs = np.full(iterations, 100000)
            prep_costs = np.full(iterations, 1500)
            st.warning("CSV Data missing or invalid. Using fallback constants (Cost=100k, Prep=1.5k).")

    # Generate Competitor Bids
    min_comp_bids = np.full(iterations, np.inf)
    comp_bid_matrix = []
    
    for comp in competitors:
        # Does the competitor bid? 1=Yes, 0=No
        bids = np.random.binomial(1, comp["prob"], iterations)
        # What is their bid if they bid?
        bid_amounts = np.random.triangular(comp["min"], comp["mode"], comp["max"], iterations)
        # If they don't bid, we consider their bid infinite so they don't win
        actual_bids = np.where(bids == 1, bid_amounts, np.inf)
        comp_bid_matrix.append(actual_bids)
        min_comp_bids = np.minimum(min_comp_bids, actual_bids)

    # Miller wins if our bid is STRICTLY LESS THAN the minimum competitor bid
    miller_wins = (bid_amount < min_comp_bids)
    
    # Calculate Profit
    # If we win: Bid - Project Cost - Prep Cost
    # If we lose: - Prep Cost
    profits = np.where(miller_wins, bid_amount - project_costs - prep_costs, -prep_costs)
    
    return profits, miller_wins, project_costs, prep_costs, min_comp_bids, comp_bid_matrix

# --- RUN MAIN SIMULATION ---
with st.spinner("Running simulation..."):
    profits, wins, proj_costs, prep_costs, min_bids, c_matrix = run_simulation(miller_bid, n_iterations)

# --- METRICS ---
expected_profit = np.mean(profits)
win_prob = np.mean(wins) * 100
std_profit = np.std(profits)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Profit", f"${expected_profit:,.2f}")
col2.metric("Probability of Winning", f"{win_prob:.1f}%")
col3.metric("Max Profit", f"${np.max(profits):,.2f}")
col4.metric("Risk (Std Dev)", f"${std_profit:,.2f}")

st.markdown("---")

# --- COST BREAKDOWN WATERFALL ---
st.subheader("💵 Financial Breakdown (If Contract is Won)")
st.markdown("This waterfall chart shows the exact dollar amounts incurred **only when Miller wins the contract**. This shows the actual un-weighted amounts changing hands.")

# Conditional values (If Won)
revenue_if_won = miller_bid
avg_proj_cost = np.mean(proj_costs)
avg_prep_cost = np.mean(prep_costs)
profit_if_won = revenue_if_won - avg_proj_cost - avg_prep_cost

fig_waterfall = go.Figure(go.Waterfall(
    name="Financials (Won)",
    orientation="v",
    measure=["relative", "relative", "relative", "total"],
    x=["Gross Revenue", "Average Project Cost", "Bid Prep Cost", "Net Profit (If Won)"],
    textposition="outside",
    text=[f"${revenue_if_won:,.0f}", f"-${avg_proj_cost:,.0f}", f"-${avg_prep_cost:,.0f}", f"${profit_if_won:,.0f}"],
    y=[revenue_if_won, -avg_proj_cost, -avg_prep_cost, profit_if_won],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    decreasing={"marker": {"color": "#e74c3c"}},
    increasing={"marker": {"color": "#2ecc71"}},
    totals={"marker": {"color": "#3498db"}}
))
fig_waterfall.update_layout(
    title="Expected Cost & Revenue Breakdown",
    showlegend=False,
    yaxis_title="Amount ($)",
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_waterfall, use_container_width=True)

st.markdown("---")

# --- VISUALIZATIONS ---
st.subheader("📊 Profit Distribution (Risk Profile)")

fig_hist = px.histogram(
    x=profits, 
    nbins=100, 
    title=f"Distribution of Profits for Bid = ${miller_bid:,}",
    labels={'x': 'Profit ($)', 'y': 'Frequency'},
    color_discrete_sequence=['#2ecc71']
)
fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
fig_hist.add_vline(x=expected_profit, line_dash="dash", line_color="blue", annotation_text="Expected Profit")
fig_hist.update_layout(showlegend=False, bargap=0.1)
st.plotly_chart(fig_hist, use_container_width=True)


st.markdown("---")
st.subheader("📈 Input Distributions (Sanity Checks)")
st.markdown("These charts allow you to verify the costs and bids used in the simulation match your expectations and CSV data.")
col_in1, col_in2 = st.columns(2)

with col_in1:
    fig_proj = px.histogram(
        x=proj_costs, 
        nbins=50, 
        title="Sampled Project Costs",
        labels={'x': 'Project Cost ($)', 'y': 'Frequency'},
        color_discrete_sequence=['#e74c3c']
    )
    fig_proj.update_layout(showlegend=False, bargap=0.1)
    st.plotly_chart(fig_proj, use_container_width=True)

with col_in2:
    fig_prep = px.histogram(
        x=prep_costs, 
        nbins=50, 
        title="Sampled Bid Preparation Costs",
        labels={'x': 'Bid Prep Cost ($)', 'y': 'Frequency'},
        color_discrete_sequence=['#f39c12']
    )
    fig_prep.update_layout(showlegend=False, bargap=0.1)
    st.plotly_chart(fig_prep, use_container_width=True)


if use_optimal_curve:
    st.markdown("---")
    st.subheader("🎯 Optimal Bid Analysis")
    st.markdown("This chart analyzes expected profit across a range of possible bids to help Miller construct the optimal strategy.")
    
    with st.spinner("Simulating bid curve..."):
        test_bids = np.arange(bid_range_min, bid_range_max + bid_range_step, bid_range_step)
        exp_profits = []
        win_probs = []
        
        for tb in test_bids:
            p, w, _, _, _, _ = run_simulation(tb, n_iterations)
            exp_profits.append(np.mean(p))
            win_probs.append(np.mean(w) * 100)
            
    best_idx = np.argmax(exp_profits)
    best_bid = test_bids[best_idx]
    best_profit = exp_profits[best_idx]
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=test_bids, y=exp_profits, mode='lines+markers', name='Expected Profit', line=dict(color='#3498db', width=3)))
    fig_curve.add_vline(x=best_bid, line_dash="dash", line_color="green", annotation_text=f"Optimal Bid: ${best_bid:,}")
    
    fig_curve.update_layout(
        title="Expected Profit vs. Bid Amount",
        xaxis_title="Bid Amount ($)",
        yaxis_title="Expected Profit ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_curve, use_container_width=True)
    
    st.success(f"**Insight:** To maximize expected profit, Miller should bid **${best_bid:,.2f}**, yielding an expected profit of **${best_profit:,.2f}**.")


# --- DATA TABLE ---
st.markdown("---")
st.subheader("📋 Sample Simulation Data (First 100 Trials)")
st.markdown("Use this raw data to double-check backend logic.")

# Construct DataFrame for the first 100 iterations
df_sample = pd.DataFrame({
    "Iteration": np.arange(1, 101),
    "Miller Bid": miller_bid,
    "Project Cost": proj_costs[:100],
    "Lowest Comp Bid": min_bids[:100]
})

for i, c_bids in enumerate(c_matrix):
    df_sample[f"Comp {chr(65+i)} Bid"] = np.where(c_bids[:100] == np.inf, "No Bid", c_bids[:100].round(2))

df_sample["Miller Wins"] = wins[:100]
df_sample["Profit"] = profits[:100]

# Formatting
st.dataframe(df_sample.style.format({
    "Miller Bid": "${:,.2f}",
    "Project Cost": "${:,.2f}",
    "Lowest Comp Bid": lambda x: f"${x:,.2f}" if x != np.inf else "No Bid",
    "Profit": "${:,.2f}"
}), use_container_width=True)

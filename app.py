import streamlit as st
import os
import sys

st.set_page_config(
    page_title="AI Real Estate Advisor",
    page_icon="🏠",
    layout="wide"
)

# ── Sidebar: API key input ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("GROQ_API_KEY", type="password",
                             help="Get free key at console.groq.com")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("API key set!")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Enter property details")
    st.markdown("2. ML model predicts price")
    st.markdown("3. AI retrieves market data")
    st.markdown("4. LLM generates advisory")

# ── Page header ────────────────────────────────────────────────────────────
st.title("🏠 AI Real Estate Investment Advisor")
st.markdown("Powered by Machine Learning + LangGraph + RAG")
st.markdown("---")

# ── Property input form ────────────────────────────────────────────────────
from agent.nodes import CITIES, STATEZIPS

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    city = st.selectbox("City", CITIES, index=CITIES.index("Seattle"))
    statezip = st.selectbox("Zip Code", STATEZIPS, index=STATEZIPS.index("WA 98103"))

with col2:
    st.subheader("🏗️ Property Details")
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0,
                                 value=2.0, step=0.25)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0,
                              value=1.0, step=0.5)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=1990)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0,
                                    max_value=2024, value=0)

with col3:
    st.subheader("📐 Size & Amenities")
    sqft_living = st.number_input("Living Area (sqft)", min_value=300,
                                   max_value=15000, value=1800)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500,
                                max_value=1500000, value=7000)
    sqft_above = st.number_input("Sqft Above Ground", min_value=300,
                                  max_value=10000, value=1500)
    sqft_basement = st.number_input("Sqft Basement", min_value=0,
                                     max_value=5000, value=300)
    waterfront = st.selectbox("Waterfront", [0, 1],
                               format_func=lambda x: "Yes" if x else "No")
    view = st.slider("View Rating", 0, 4, 0)
    condition = st.slider("Condition Rating", 1, 5, 3)

st.markdown("---")

# ── Run button ─────────────────────────────────────────────────────────────
if st.button("🔍 Analyze Property & Generate Advisory", use_container_width=True,
             type="primary"):

    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar first!")
    else:
        from agent.graph import build_graph

        property_input = {
            "bedrooms": float(bedrooms),
            "bathrooms": float(bathrooms),
            "sqft_living": float(sqft_living),
            "sqft_lot": float(sqft_lot),
            "floors": float(floors),
            "waterfront": int(waterfront),
            "view": int(view),
            "condition": int(condition),
            "sqft_above": float(sqft_above),
            "sqft_basement": float(sqft_basement),
            "yr_built": int(yr_built),
            "yr_renovated": int(yr_renovated),
            "city": city,
            "statezip": statezip,
            "predicted_price": None,
            "market_context": None,
            "advisory_report": None,
            "error": None
        }

        with st.spinner("Step 1/3: Running ML price prediction..."):
            graph = build_graph()

        progress = st.progress(0)

        with st.spinner("Step 2/3: Retrieving market insights from knowledge base..."):
            progress.progress(33)

        with st.spinner("Step 3/3: Generating investment advisory with AI..."):
            result = graph.invoke(property_input)
            progress.progress(100)

        # ── Results ────────────────────────────────────────────────────────
        st.success("Analysis complete!")
        st.markdown("---")

        # Price prediction
        if result.get("predicted_price"):
            price = result["predicted_price"]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Predicted Price", f"${price:,.0f}")
            col_b.metric("Location", f"{city}")
            col_c.metric("Size", f"{sqft_living:,} sqft")

        st.markdown("---")

        # Advisory report
        if result.get("advisory_report"):
            st.subheader("📋 Investment Advisory Report")
            st.markdown(result["advisory_report"])

        # Show what was retrieved from RAG (expandable)
        if result.get("market_context"):
            with st.expander("🔍 Market Data Retrieved (RAG)"):
                st.text(result["market_context"])

        if result.get("error"):
            st.error(f"Error: {result['error']}")
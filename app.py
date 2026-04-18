import streamlit as st
import os

st.set_page_config(
    page_title="AI Real Estate Advisor",
    page_icon="🏠",
    layout="wide"
)


# ── Header ────────────────────────────────────────────────────────────────
st.title("🏠 AI Real Estate Investment Advisor")
st.markdown("ML + RAG + LangGraph + LLM")
st.markdown("---")

# ── Inputs ────────────────────────────────────────────────────────────────
from agent.nodes import CITIES, CITY_ZIP_MAP

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    city = st.selectbox("City", CITIES, index=CITIES.index("Seattle"))

    # ✅ FIXED: ZIP depends on city
    zip_options = CITY_ZIP_MAP.get(city, [])

    area = st.selectbox(
        "Area / Zip Code",
        zip_options,
        index=0
    )

with col2:
    st.subheader("🏗️ Property Details")
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0, step=0.25)
    floors = st.number_input("Floors", 1.0, 4.0, 1.0, step=0.5)
    yr_built = st.number_input("Year Built", 1900, 2024, 1990)
    yr_renovated = st.number_input("Year Renovated", 0, 2024, 0)

with col3:
    st.subheader("📐 Size & Amenities")
    sqft_living = st.number_input("Living Area", 300, 15000, 1800)
    sqft_lot = st.number_input("Lot Size", 500, 1500000, 7000)
    sqft_above = st.number_input("Sqft Above", 300, 10000, 1500)
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 300)
    waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x else "No")
    view = st.slider("View", 0, 4, 0)
    condition = st.slider("Condition", 1, 5, 3)

st.markdown("---")

# ── Run Analysis ───────────────────────────────────────────────────────────
if st.button("🔍 Analyze Property", use_container_width=True):

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
            "statezip": area,
            "predicted_price": None,
            "market_context": None,
            "advisory_report": None,
            "error": None
        }

        graph = build_graph()
        result = graph.invoke(property_input)

        st.session_state["result"] = result
        st.session_state["property"] = property_input

        st.success("Analysis complete!")

        if result.get("predicted_price"):
            price = result["predicted_price"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${price:,.0f}")
            col2.metric("City", city)
            col3.metric("Size", f"{sqft_living} sqft")

        st.markdown("---")
        st.subheader("📋 Advisory Report")
        st.markdown(result.get("advisory_report", ""))

        with st.expander("🔍 Market Data"):
            st.text(result.get("market_context", ""))

        if result.get("error"):
            st.error(result["error"])

# ── Chat Section ───────────────────────────────────────────────────────────
if "result" in st.session_state:

    st.markdown("---")
    st.subheader("💬 Ask Follow-up Questions")

    user_query = st.text_input("💬 Ask anything about this property")

    if user_query:
        banned_keywords = [
            "murder", "kill", "crime", "weapon", "hack", "drugs",
            "violence", "attack", "bomb"
        ]

        if any(word in user_query.lower() for word in banned_keywords):
            st.warning("❌ I can only assist with real estate and property-related queries.")
        else:
            from rag.retriever import retrieve_context
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=st.secrets["GROQ_API_KEY"]
                    )

            context = retrieve_context(user_query, selected_city)

            prompt = f"""
You are a professional real estate advisor.

PROPERTY:
{st.session_state["property"]}

MARKET CONTEXT:
{context}

USER QUESTION:
{user_query}

Answer clearly and professionally.
"""

            response = llm.invoke(prompt)

            st.markdown("### 🤖 AI Response")
            st.write(response.content)
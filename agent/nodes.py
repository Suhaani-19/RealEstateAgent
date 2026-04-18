import joblib
import pandas as pd
import numpy as np
from agent.state import PropertyState
from rag.retriever import retrieve_context

# Load your Milestone 1 model once when the file loads
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = model.feature_names_in_

CITY_ZIP_MAP = {
    "Algona": ["WA 98001"],
    "Auburn": ["WA 98001", "WA 98002", "WA 98092"],
    "Beaux Arts Village": ["WA 98004"],
    "Bellevue": ["WA 98004", "WA 98005", "WA 98006", "WA 98007", "WA 98008"],
    "Black Diamond": ["WA 98010"],
    "Bothell": ["WA 98011", "WA 98021", "WA 98041"],
    "Burien": ["WA 98166"],
    "Carnation": ["WA 98014"],
    "Clyde Hill": ["WA 98004"],
    "Covington": ["WA 98042"],
    "Des Moines": ["WA 98198"],
    "Duvall": ["WA 98019"],
    "Enumclaw": ["WA 98022"],
    "Fall City": ["WA 98024"],
    "Federal Way": ["WA 98003", "WA 98023"],
    "Inglewood-Finn Hill": ["WA 98034"],
    "Issaquah": ["WA 98027", "WA 98029"],
    "Kenmore": ["WA 98028"],
    "Kent": ["WA 98030", "WA 98031", "WA 98032"],
    "Kirkland": ["WA 98033", "WA 98034"],
    "Lake Forest Park": ["WA 98155"],
    "Maple Valley": ["WA 98038"],
    "Medina": ["WA 98039"],
    "Mercer Island": ["WA 98040"],
    "Milton": ["WA 98354"],
    "Newcastle": ["WA 98056"],
    "Normandy Park": ["WA 98166"],
    "North Bend": ["WA 98045"],
    "Pacific": ["WA 98047"],
    "Preston": ["WA 98050"],
    "Ravensdale": ["WA 98051"],
    "Redmond": ["WA 98052", "WA 98053"],
    "Renton": ["WA 98055", "WA 98057", "WA 98058", "WA 98059"],
    "Sammamish": ["WA 98074", "WA 98075"],
    "SeaTac": ["WA 98188"],
    "Seattle": [
        "WA 98101", "WA 98102", "WA 98103", "WA 98104", "WA 98105",
        "WA 98106", "WA 98107", "WA 98108", "WA 98109", "WA 98112",
        "WA 98115", "WA 98116", "WA 98117", "WA 98118", "WA 98119",
        "WA 98122", "WA 98125", "WA 98126", "WA 98133", "WA 98136",
        "WA 98144", "WA 98146", "WA 98148", "WA 98155", "WA 98166",
        "WA 98168", "WA 98177", "WA 98178", "WA 98188", "WA 98198", "WA 98199"
    ],
    "Shoreline": ["WA 98133", "WA 98155"],
    "Skykomish": ["WA 98288"],
    "Snoqualmie": ["WA 98065"],
    "Snoqualmie Pass": ["WA 98068"],
    "Tukwila": ["WA 98168"],
    "Vashon": ["WA 98070"],
    "Woodinville": ["WA 98072"],
    "Yarrow Point": ["WA 98004"]
}

CITIES = ['Algona', 'Auburn', 'Beaux Arts Village', 'Bellevue', 'Black Diamond',
          'Bothell', 'Burien', 'Carnation', 'Clyde Hill', 'Covington', 'Des Moines',
          'Duvall', 'Enumclaw', 'Fall City', 'Federal Way', 'Inglewood-Finn Hill',
          'Issaquah', 'Kenmore', 'Kent', 'Kirkland', 'Lake Forest Park',
          'Maple Valley', 'Medina', 'Mercer Island', 'Milton', 'Newcastle',
          'Normandy Park', 'North Bend', 'Pacific', 'Preston', 'Ravensdale',
          'Redmond', 'Renton', 'Sammamish', 'SeaTac', 'Seattle', 'Shoreline',
          'Skykomish', 'Snoqualmie', 'Snoqualmie Pass', 'Tukwila', 'Vashon',
          'Woodinville', 'Yarrow Point']

STATEZIPS = ['WA 98001','WA 98002','WA 98003','WA 98004','WA 98005','WA 98006',
             'WA 98007','WA 98008','WA 98010','WA 98011','WA 98014','WA 98019',
             'WA 98022','WA 98023','WA 98024','WA 98027','WA 98028','WA 98029',
             'WA 98030','WA 98031','WA 98032','WA 98033','WA 98034','WA 98038',
             'WA 98039','WA 98040','WA 98042','WA 98045','WA 98047','WA 98050',
             'WA 98051','WA 98052','WA 98053','WA 98055','WA 98056','WA 98057',
             'WA 98058','WA 98059','WA 98065','WA 98068','WA 98070','WA 98072',
             'WA 98074','WA 98075','WA 98077','WA 98092','WA 98102','WA 98103',
             'WA 98105','WA 98106','WA 98107','WA 98108','WA 98109','WA 98112',
             'WA 98115','WA 98116','WA 98117','WA 98118','WA 98119','WA 98122',
             'WA 98125','WA 98126','WA 98133','WA 98136','WA 98144','WA 98146',
             'WA 98148','WA 98155','WA 98166','WA 98168','WA 98177','WA 98178',
             'WA 98188','WA 98198','WA 98199','WA 98288','WA 98354']


def node_predict_price(state: PropertyState) -> PropertyState:
    """Node 1: Uses Milestone 1 model to predict price."""
    try:
        zip_value = state.get("statezip", "")

        input_data = {
            'bedrooms': [state['bedrooms']],
            'bathrooms': [state['bathrooms']],
            'sqft_living': [state['sqft_living']],
            'sqft_lot': [state['sqft_lot']],
            'floors': [state['floors']],
            'waterfront': [state['waterfront']],
            'view': [state['view']],
            'condition': [state['condition']],
            'sqft_above': [state['sqft_above']],
            'sqft_basement': [state['sqft_basement']],
            'yr_built': [state['yr_built']],
            'yr_renovated': [state['yr_renovated']],
        }

        # One-hot encode city
        for city in CITIES:
            input_data[f'city_{city}'] = [1 if state['city'] == city else 0]

        # One-hot encode statezip (SAFE VERSION)
        for sz in STATEZIPS:
            input_data[f'statezip_{sz}'] = [1 if zip_value == sz else 0]

        df_input = pd.DataFrame(input_data)

        # Ensure all model columns exist
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[model_columns]

        # Scale numeric features safely
        numeric_cols = ['sqft_living','sqft_lot','sqft_above','sqft_basement']
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        price = float(model.predict(df_input)[0])

        return {**state, "predicted_price": price, "error": None}

    except Exception as e:
        return {**state, "predicted_price": None, "error": str(e)}


def node_retrieve_market(state: PropertyState) -> PropertyState:
    try:
        query = f"{state['city']} {state['statezip']} real estate investment property"
        context = retrieve_context(query, state["city"])
        return {**state, "market_context": context}
    except Exception as e:
        return {**state, "market_context": "Market data unavailable.", "error": str(e)}


def node_generate_advisory(state: PropertyState) -> PropertyState:
    """Node 3: LLM advisory generation."""
    try:
        from langchain_groq import ChatGroq
        import os
        import streamlit as st
        

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=st.secrets["GROQ_API_KEY"]
        )

        price = state.get("predicted_price")

        if price is None:
            return {**state, "advisory_report": "Price prediction failed."}

        context = state.get("market_context", "")

        prompt = f"""
You are a real estate advisor.

PROPERTY:
- City: {state['city']}
- ZIP: {state['statezip']}
- Bedrooms: {state['bedrooms']}
- Bathrooms: {state['bathrooms']}
- Sqft: {state['sqft_living']}

Predicted Price: ${price:,.0f}

Market Context:
{context}

Give:
1. Price Analysis
2. Market Trend
3. BUY / HOLD / SELL
4. Risks
5. Outlook

Be concise and professional.
"""

        response = llm.invoke(prompt)

        return {**state, "advisory_report": response.content}

    except Exception as e:
        return {**state, "advisory_report": f"Error: {str(e)}"}
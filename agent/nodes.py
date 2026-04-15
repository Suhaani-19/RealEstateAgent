import joblib
import pandas as pd
import numpy as np
from agent.state import PropertyState
from rag.retriever import retrieve_context

# Load your Milestone 1 model once when the file loads
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = model.feature_names_in_

# These are the exact cities and statezips from your dataset
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
    """Node 1: Uses your Milestone 1 Linear Regression model to predict price."""
    try:
        # Build a dataframe exactly the way your Colab notebook did
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

        # One-hot encode statezip
        for sz in STATEZIPS:
            input_data[f'statezip_{sz}'] = [1 if state['statezip'] == sz else 0]

        df_input = pd.DataFrame(input_data)
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[model_columns]
        # Scale numeric features
        numeric_cols = ['sqft_living','sqft_lot','sqft_above','sqft_basement']
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        price = float(model.predict(df_input)[0])
        return {**state, "predicted_price": price, "error": None}

    except Exception as e:
        return {**state, "predicted_price": None, "error": str(e)}


def node_retrieve_market(state: PropertyState) -> PropertyState:
    """Node 2: Retrieves relevant market knowledge from your RAG vector DB."""
    try:
        query = f"{state['city']} {state['statezip']} real estate investment property"
        context = retrieve_context(query)
        return {**state, "market_context": context}
    except Exception as e:
        return {**state, "market_context": "Market data unavailable.", "error": str(e)}


def node_generate_advisory(state: PropertyState) -> PropertyState:
    """Node 3: Calls Groq LLM to generate a structured investment report."""
    try:
        from langchain_groq import ChatGroq
        import os

        llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY", "")
        )

        price = state.get("predicted_price")

        if price is None:
            return {**state, "advisory_report": "Price prediction failed. Cannot generate advisory."}
        context = state.get("market_context", "")

        prompt = f"""You are a professional real estate investment advisor specializing in King County, Washington.

PROPERTY DETAILS:
- Location: {state['city']}, {state['statezip']}
- Bedrooms: {state['bedrooms']} | Bathrooms: {state['bathrooms']}
- Living Area: {state['sqft_living']:,.0f} sqft | Lot: {state['sqft_lot']:,.0f} sqft
- Floors: {state['floors']} | Waterfront: {'Yes' if state['waterfront'] else 'No'}
- View Rating: {state['view']}/4 | Condition: {state['condition']}/5
- Year Built: {state['yr_built']} | Year Renovated: {state['yr_renovated'] if state['yr_renovated'] > 0 else 'Never'}

ML PREDICTED PRICE: ${price:,.0f}

MARKET KNOWLEDGE:
{context}

Write a structured investment advisory report with these exact sections:

1. PRICE ASSESSMENT
Explain if this property is fairly priced, undervalued, or overvalued based on the prediction and market knowledge.

2. MARKET TREND ANALYSIS  
Summarize what the market data says about this location and property type.

3. INVESTMENT RECOMMENDATION
Give a clear BUY / HOLD / AVOID recommendation with specific reasoning.

4. RISK FACTORS
List 3 specific risks for this property.

5. 2-YEAR OUTLOOK
Give a realistic outlook for this property's value and rental potential.

DISCLAIMER: This is AI-generated analysis for educational purposes only, not professional financial advice.
"""

        response = llm.invoke(prompt)
        return {**state, "advisory_report": response.content}

    except Exception as e:
        return {**state, "advisory_report": f"Could not generate advisory: {str(e)}"}
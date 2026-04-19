#  AI Real Estate Investment Advisor 

An intelligent real estate advisory system that combines **Machine Learning**, **Retrieval-Augmented Generation (RAG)**, and **LLMs** to provide data-driven investment recommendations for properties in King County, Washington.

##  Overview

The AI Real Estate Investment Advisor helps users make informed decisions about real estate investments by:

1. **Predicting Property Prices** - ML model estimates fair market value based on property features
2. **Retrieving Market Context** - RAG system fetches relevant market data and trends from a knowledge base
3. **Generating Advisory Reports** - LLM creates personalized investment recommendations
4. **Enabling Follow-up Questions** - Interactive chat interface for deeper exploration

### Technology Stack

- **Streamlit** - Interactive web UI
- **LangGraph** - Agentic workflow orchestration
- **LangChain** - LLM framework and RAG pipeline
- **Groq LLM** - Fast inference for advisory generation
- **ChromaDB** - Vector database for semantic search
- **Scikit-learn** - Machine learning for price prediction
- **HuggingFace Embeddings** - Semantic text embeddings

##  Project Structure

```
RealEstateAgent/
├── app.py                          # Streamlit frontend application
├── train.py                        # ML model training script
├── data.csv                        # Property dataset for training
├── requirements.txt                # Python dependencies
├── house_price_model.pkl           # Trained ML model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── model_columns.pkl               # Model feature columns (generated)
├── chroma_db/                      # Vector database (generated)
│
├── agent/                          # Agentic workflow logic
│   ├── state.py                    # PropertyState TypedDict definition
│   ├── nodes.py                    # Workflow nodes (price prediction, RAG, advisory)
│   └── graph.py                    # LangGraph workflow orchestration
│
├── rag/                            # Retrieval-Augmented Generation
│   ├── build_index.py              # Script to build vector index from knowledge base
│   └── retriever.py                # Vector database retrieval interface
│
└── knowledge_base/                 # Domain knowledge documents
    ├── investment_guide.txt        # Real estate investment guidelines
    ├── market_trends.txt           # King County market data
    └── neighborhoods.txt           # Neighborhood-specific information
```

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- GROQ API key ([Get one here](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   cd RealEstateAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the ML model**
   ```bash
   python train.py
   ```
   This trains and saves the house price prediction model.

4. **Build the RAG index** (if not already present)
   ```bash
   python rag/build_index.py
   ```
   This creates a vector database from the knowledge base documents.

### Running the Application

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and enter your GROQ API key in the sidebar.

##  Features

### 1. Property Input
Select or enter property details:
- **Location** - City and ZIP code (King County area)
- **Property Features** - Bedrooms, bathrooms, square footage, etc.
- **Condition** - Building condition rating (1-5)
- **Special Features** - Waterfront status, view rating

### 2. ML Price Prediction
- Trained on historical property data
- Normalizes features using StandardScaler
- Outputs predicted market value
- Helps identify overpriced/underpriced properties

### 3. Market Context (RAG)
- Retrieves relevant market trends and investment guidelines
- Uses semantic search on knowledge base
- Provides context like:
  - Average prices by neighborhood
  - Historical appreciation rates
  - Investment signals and risk factors
  - Rental yield information

### 4. AI Investment Advisory
- Uses Groq LLM for fast inference
- Generates personalized investment recommendations
- Considers ML prediction vs. asking price
- Provides BUY/HOLD/AVOID signals
- Suggests investment strategies

### 5. Interactive Chat
- Ask follow-up questions about the property
- Get deeper insights on specific aspects
- Explore alternative scenarios

## Workflow

```
User Input (Property Details)
    ↓
ML Price Prediction → Predicted Price
    ↓
RAG Retrieval → Market Context
    ↓
LLM Generation → Investment Advisory Report
    ↓
Display Results + Enable Chat
```

## Machine Learning Model

### Training Process (train.py)

1. Loads `data.csv` with historical property data
2. Cleans data (drops non-numeric columns, handles missing values)
3. Encodes categorical features with one-hot encoding
4. Scales features using StandardScaler
5. Trains LinearRegression model
6. Saves model artifacts:
   - `house_price_model.pkl` - Trained model
   - `scaler.pkl` - Feature scaler
   - `model_columns.pkl` - Feature column names

### Model Inference (nodes.py)

The trained model is loaded once at startup and reused for predictions across multiple requests.

## RAG System

### Knowledge Base

Located in `knowledge_base/`:
- **investment_guide.txt** - Investment signals, risk factors, ROI calculations
- **market_trends.txt** - King County market data, appreciation rates
- **neighborhoods.txt** - Neighborhood-specific information

### Retrieval Process (retriever.py)

1. Uses HuggingFace embeddings (`all-MiniLM-L6-v2`)
2. Stores documents in ChromaDB vector database
3. On query: Performs semantic similarity search (top-4 results by default)
4. Returns concatenated relevant context to LLM

##  LangGraph Workflow

The agent workflow (`agent/graph.py`) orchestrates:
1. **Predict Node** - Calls ML model for price prediction
2. **Retrieve Node** - Fetches market context via RAG
3. **Generate Node** - Creates advisory report using LLM

State flows through these nodes with error handling and result compilation.

##  User Interface

Built with Streamlit:
- **Sidebar** - API key configuration and workflow explanation
- **Main Area** - Property input form and results
- **Results Panel** - Displays predicted price, market context, advisory
- **Chat Interface** - Interactive follow-up questions

## Configuration

### Required Environment Variables

- `GROQ_API_KEY` - Your Groq API key (set via Streamlit sidebar)

### Optional Settings

- Model choice can be configured in LLM nodes
- Retrieval parameters (k=4 for similarity search) in `retriever.py`
- Embedding model in `retriever.py` and `build_index.py`

##  Performance Considerations

- **Model Loading** - Loaded once at startup for efficiency
- **Embeddings** - Cached in ChromaDB for fast retrieval
- **LLM Inference** - Uses Groq for fast response times
- **Streamlit Caching** - Can be added for frequently accessed data

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| API key not recognized | Verify GROQ_API_KEY is valid and has proper permissions |
| Model not found | Run `python train.py` to train and save the model |
| ChromaDB not found | Run `python rag/build_index.py` to build the vector index |
| Embedding errors | Ensure `sentence-transformers` is installed |
| City/ZIP not found | Check `agent/nodes.py` for available CITY_ZIP_MAP entries |

##  Available Locations

The system supports 39 cities in King County, Washington, with over 100 ZIP codes including:
- Seattle (10 ZIP codes)
- Bellevue (5 ZIP codes)
- Redmond (2 ZIP codes)
- And 36 other King County cities

## Future Enhancements

- [ ] Real-time MLS data integration
- [ ] Multi-region support beyond King County
- [ ] Investment portfolio analysis
- [ ] Rental property analysis tools
- [ ] Price trend forecasting
- [ ] Neighborhood walkability scores
- [ ] School district ratings

## Streamlit Link 
https://realestateagent.streamlit.app/
      

## Team

| Name | Enrollment Number | Role |
|------|-------------------|------|
| Suhaani Garg | 2401010462 | Core AI (ML + RAG + LLM, LangGraph) |
| Prachee Dhar | 2401010330 | Documentation & Report |
| Manjeet | 2401010262 | Github & Streamlit UI & Deployment Setup |
| Aarya Srivastava | 2401010008 | UI & Frontend Integration |

##  License

This project is open source and available under the MIT License.

---

**Note**: This system provides data-driven insights but should not be the sole basis for investment decisions. Always consult with qualified real estate professionals and financial advisors.

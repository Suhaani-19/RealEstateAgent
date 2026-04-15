from typing import TypedDict, Optional

class PropertyState(TypedDict):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    city: str
    statezip: str
    predicted_price: Optional[float]
    market_context: Optional[str]
    advisory_report: Optional[str]
    error: Optional[str]
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PriceBand:
    id: str
    name: str
    action: str
    min_price: Optional[float]
    max_price: Optional[float]
    color: str

    @property
    def range_label(self) -> str:
        if self.min_price is None:
            return "< $0"
        elif self.max_price is None:
            return f"> ${self.min_price:,.0f}"
        else:
            lo = f"${self.min_price:,.0f}"
            hi = f"${self.max_price:,.0f}"
            return f"{lo} – {hi}"

    def contains(self, price: float) -> bool:
        lower_ok = self.min_price is None or price >= self.min_price
        upper_ok = self.max_price is None or price < self.max_price
        return lower_ok and upper_ok


STORAGE_BANDS: list[PriceBand] = [
    PriceBand("B1", "Negative",  "Charge Maximum",      None,  0,     "#166534"),
    PriceBand("B2", "Low",       "Charge",              0,     50,    "#16a34a"),
    PriceBand("B3", "Neutral",   "Hold",                50,    150,   "#ca8a04"),
    PriceBand("B4", "Elevated",  "Consider Discharge",  150,   300,   "#ea580c"),
    PriceBand("B5", "High",      "Discharge",           300,   1000,  "#dc2626"),
    PriceBand("B6", "Spike",     "Peak Revenue",        1000,  5000,  "#9f1239"),
    PriceBand("B7", "Extreme",   "Max Revenue Event",   5000,  None,  "#6b21a8"),
]

import pandas as pd
import pytest
from inventory_intelligence.data import load_inputs, DataValidationError

def test_rejects_invalid_sales_date():
    sales = pd.DataFrame([{"date": "not-a-date", "product_id": "A", "units_sold": 1}])
    inventory = pd.DataFrame([{"product_id": "A", "on_hand": 2}])
    products = pd.DataFrame([{"product_id": "A"}])
    suppliers = pd.DataFrame([{"product_id": "A", "lead_time_days": 3}])
    with pytest.raises(DataValidationError):
        load_inputs(sales, inventory, products, suppliers)

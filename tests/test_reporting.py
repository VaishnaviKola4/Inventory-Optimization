from inventory_intelligence.reporting import build_pdf_report

def test_pdf_report_returns_pdf_bytes():
    result={
      "generated_at":"2026-10-23T12:00:00+05:30","horizon_days":30,"service_level":0.95,
      "summary":{"products_evaluated":1,"reorder_count":1,"hold_count":0,"promote_count":0},
      "inventory_recommendations":[{
        "product_id":"SKU-1","recommendation_type":"Reorder","quantity":20,
        "timing":"Order immediately","reasoning":"Position at reorder point.",
        "forecast_demand_units":50,"supplier_lead_time_days":5,"metrics":{}
      }]
    }
    pdf=build_pdf_report(result)
    assert pdf.startswith(b"%PDF")
    assert len(pdf)>500

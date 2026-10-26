\
from __future__ import annotations
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

NAVY = colors.HexColor("#232F3E")
ACCENT = colors.HexColor("#FF9900")
PALE = colors.HexColor("#F3F5F7")
SLATE = colors.HexColor("#485465")

def build_pdf_report(result: dict) -> bytes:
    """Build an executive-style, downloadable inventory report in memory."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=12*mm,
                            leftMargin=12*mm, topMargin=13*mm, bottomMargin=15*mm,
                            title="AI Inventory Intelligence Report", author="AI Inventory Intelligence")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], textColor=NAVY,
                              fontSize=22, leading=26, alignment=TA_LEFT, spaceAfter=5))
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], textColor=SLATE,
                              fontSize=8.5, leading=11, spaceAfter=7))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=NAVY,
                              fontSize=12, leading=15, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallCell", parent=styles["BodyText"], fontSize=7.2, leading=8.5))
    story=[]
    generated=result.get("generated_at", datetime.now().astimezone().isoformat(timespec="seconds"))
    summary=result.get("summary", {})
    recs=result.get("inventory_recommendations", [])
    story.append(Paragraph("AI Inventory Intelligence", styles["ReportTitle"]))
    story.append(Paragraph(f"Inventory decision-support report | Generated {generated} | Forecast horizon: {result.get('horizon_days','N/A')} days | Service level: {result.get('service_level','N/A')}", styles["Subtle"]))
    summary_data=[["Products evaluated","Reorder actions","Hold actions","Promote reviews"],
                  [str(summary.get("products_evaluated",0)),str(summary.get("reorder_count",0)),
                   str(summary.get("hold_count",0)),str(summary.get("promote_count",0))]]
    cards=Table(summary_data,colWidths=[60*mm]*4,rowHeights=[8*mm,11*mm])
    cards.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("BACKGROUND",(0,1),(-1,1),PALE),("TEXTCOLOR",(0,1),(-1,1),NAVY),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8.5),("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BOX",(0,0),(-1,-1),0.5,colors.lightgrey),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.lightgrey)]))
    story.extend([cards,Spacer(1,5*mm),Paragraph("Prioritized inventory recommendations",styles["SectionTitle"])])
    headers=["Product","Warehouse","Action","Quantity","Timing","Forecast units","Lead time","Evidence"]
    rows=[headers]
    for r in recs:
        reason=str(r.get("reasoning",""))
        if len(reason)>170: reason=reason[:167]+"..."
        rows.append([Paragraph(str(r.get("product_id","")),styles["SmallCell"]),
                     Paragraph(str(r.get("warehouse_id") or "—"),styles["SmallCell"]),
                     Paragraph(str(r.get("recommendation_type","")),styles["SmallCell"]),
                     str(r.get("quantity",0)),Paragraph(str(r.get("timing","")),styles["SmallCell"]),
                     str(r.get("forecast_demand_units",0)),str(r.get("supplier_lead_time_days","")),
                     Paragraph(reason,styles["SmallCell"])])
    if len(rows)==1: rows.append(["—","—","—","—","No recommendations","—","—","—"])
    table=Table(rows,colWidths=[22*mm,18*mm,22*mm,16*mm,31*mm,22*mm,17*mm,88*mm],repeatRows=1)
    cmds=[("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),7.5),
          ("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),0.35,colors.lightgrey),
          ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
          ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]
    for i in range(1,len(rows)):
        if i%2==0: cmds.append(("BACKGROUND",(0,i),(-1,i),PALE))
    table.setStyle(TableStyle(cmds)); story.append(table)
    story.extend([Spacer(1,4*mm),Paragraph("Governance and interpretation",styles["SectionTitle"]),
      Paragraph("This report supports decisions; it does not create purchase orders. Validate recommendations against actual supplier calendars, order multiples, product expiry, and inventory freshness. Forecast performance should be assessed with time-based backtesting. A negative Promote quantity is an approximate excess quantity for markdown/clearance review, not an instruction to dispose of stock.",styles["Subtle"])])
    def footer(canvas, document):
        canvas.saveState(); w,h=landscape(A4); canvas.setStrokeColor(ACCENT); canvas.setLineWidth(1.4)
        canvas.line(12*mm,10*mm,w-12*mm,10*mm); canvas.setFont("Helvetica",7.5); canvas.setFillColor(SLATE)
        canvas.drawString(12*mm,6*mm,"AI Inventory Intelligence | Human review required")
        canvas.drawRightString(w-12*mm,6*mm,f"Page {document.page}"); canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()

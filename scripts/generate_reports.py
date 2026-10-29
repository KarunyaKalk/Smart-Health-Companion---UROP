import csv
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Synthetic research/demo data only. These patterns are NOT clinical rules.
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "synthetic_reports"
LABEL_DIR = BASE_DIR / "data" / "ground_truth"
PDF_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_COUNT = 2000  # Change to 2500, 3000, etc. if needed.
SEED = 42
random.seed(SEED)

CLASSES = [
    "no_target_detected",
    "dengue",
    "malaria",
    "typhoid_fever",
    "tuberculosis",
    "covid_19",
    "hepatitis_b",
]
LAYOUTS = ["classic", "compact", "minimal"]

# Feature specifications. Numeric ranges are illustrative synthetic ranges,
# NOT clinical reference ranges or diagnostic thresholds.
NUMERIC = {
    "fever_duration_days": (0, 14, 0),
    "temperature_c": (36.2, 40.5, 1),
    "wbc_count": (2500, 16000, 0),
    "platelet_count": (50000, 450000, 0),
    "hemoglobin": (8.0, 17.5, 1),
    "neutrophil_percent": (25, 85, 1),
    "lymphocyte_percent": (10, 60, 1),
    "crp_mg_l": (0, 180, 1),
    "procalcitonin_ng_ml": (0.02, 8.0, 2),
    "oxygen_saturation_percent": (85, 100, 1),
    "cough_duration_days": (0, 60, 0),
}
TEST_FEATURES = [
    "dengue_ns1_result",
    "dengue_igm_result",
    "malaria_antigen_result",
    "malaria_smear_result",
    "typhoid_blood_culture",
    "tb_gene_xpert_result",
    "sputum_afb_smear_result",
    "covid_pcr_result",
    "hepatitis_b_hbsag_result",
]
FEATURES = list(NUMERIC.keys()) + TEST_FEATURES
TARGET = "target_pattern"

TEST_NAMES = {
    "dengue_ns1_result": "Dengue NS1",
    "dengue_igm_result": "Dengue IgM",
    "malaria_antigen_result": "Malaria antigen",
    "malaria_smear_result": "Malaria blood smear",
    "typhoid_blood_culture": "Typhoid blood culture",
    "tb_gene_xpert_result": "TB molecular test (GeneXpert)",
    "sputum_afb_smear_result": "Sputum AFB smear",
    "covid_pcr_result": "SARS-CoV-2 PCR",
    "hepatitis_b_hbsag_result": "Hepatitis B HBsAg",
}

def num(key, low=None, high=None):
    minimum, maximum, digits = NUMERIC[key]
    value = random.uniform(minimum if low is None else low,
                           maximum if high is None else high)
    return round(value, digits)

def categorical_test_results(label):
    # Not-performed is distinct from negative. Relevant tests are simulated
    # positive sometimes, negative sometimes to avoid a perfect label shortcut.
    results = {key: "NOT_PERFORMED" for key in TEST_FEATURES}
    disease_tests = {
        "dengue": ["dengue_ns1_result", "dengue_igm_result"],
        "malaria": ["malaria_antigen_result", "malaria_smear_result"],
        "typhoid_fever": ["typhoid_blood_culture"],
        "tuberculosis": ["tb_gene_xpert_result", "sputum_afb_smear_result"],
        "covid_19": ["covid_pcr_result"],
        "hepatitis_b": ["hepatitis_b_hbsag_result"],
    }
    if label in disease_tests:
        for key in disease_tests[label]:
            results[key] = random.choices(
                ["POSITIVE", "NEGATIVE", "INDETERMINATE"],
                weights=[0.82, 0.15, 0.03],
                k=1,
            )[0]
    else:
        # For baseline examples, selected tests may have been performed and negative.
        for key in random.sample(TEST_FEATURES, k=random.randint(1, 3)):
            results[key] = random.choices(
                ["NEGATIVE", "NOT_PERFORMED", "INDETERMINATE"],
                weights=[0.75, 0.20, 0.05],
                k=1,
            )[0]
    return results

def generate_features(label):
    # Start with generic plausible-looking synthetic values. They are not
    # intended to mimic clinical distributions or guide diagnosis.
    v = {key: num(key) for key in NUMERIC}

    # Simulation-only patterns. Labels are assigned by generator, not by a clinician.
    if label == "no_target_detected":
        v.update({
            "fever_duration_days": random.randint(0, 2),
            "temperature_c": num("temperature_c", 36.2, 37.8),
            "wbc_count": num("wbc_count", 4500, 10500),
            "platelet_count": num("platelet_count", 160000, 400000),
            "crp_mg_l": num("crp_mg_l", 0, 12),
            "procalcitonin_ng_ml": num("procalcitonin_ng_ml", 0.02, 0.15),
            "oxygen_saturation_percent": num("oxygen_saturation_percent", 96, 100),
            "cough_duration_days": random.randint(0, 4),
        })
    elif label == "dengue":
        v.update({
            "fever_duration_days": random.randint(2, 7),
            "temperature_c": num("temperature_c", 38.0, 40.2),
            "wbc_count": num("wbc_count", 2500, 6500),
            "platelet_count": num("platelet_count", 50000, 150000),
            "crp_mg_l": num("crp_mg_l", 5, 60),
            "cough_duration_days": random.randint(0, 3),
        })
    elif label == "malaria":
        v.update({
            "fever_duration_days": random.randint(2, 12),
            "temperature_c": num("temperature_c", 38.0, 40.5),
            "wbc_count": num("wbc_count", 3000, 11000),
            "platelet_count": num("platelet_count", 60000, 180000),
            "crp_mg_l": num("crp_mg_l", 10, 100),
        })
    elif label == "typhoid_fever":
        v.update({
            "fever_duration_days": random.randint(4, 14),
            "temperature_c": num("temperature_c", 38.0, 40.0),
            "wbc_count": num("wbc_count", 2500, 8000),
            "crp_mg_l": num("crp_mg_l", 15, 120),
            "procalcitonin_ng_ml": num("procalcitonin_ng_ml", 0.1, 2.0),
        })
    elif label == "tuberculosis":
        v.update({
            "fever_duration_days": random.randint(3, 14),
            "temperature_c": num("temperature_c", 37.5, 39.0),
            "hemoglobin": num("hemoglobin", 8.5, 13.0),
            "lymphocyte_percent": num("lymphocyte_percent", 20, 50),
            "crp_mg_l": num("crp_mg_l", 15, 100),
            "oxygen_saturation_percent": num("oxygen_saturation_percent", 90, 98),
            "cough_duration_days": random.randint(14, 60),
        })
    elif label == "covid_19":
        v.update({
            "fever_duration_days": random.randint(1, 10),
            "temperature_c": num("temperature_c", 37.5, 39.5),
            "lymphocyte_percent": num("lymphocyte_percent", 10, 35),
            "crp_mg_l": num("crp_mg_l", 5, 100),
            "oxygen_saturation_percent": num("oxygen_saturation_percent", 88, 99),
            "cough_duration_days": random.randint(2, 30),
        })
    elif label == "hepatitis_b":
        # Many HBV infections may be asymptomatic; these are synthetic examples only.
        v.update({
            "fever_duration_days": random.randint(0, 3),
            "temperature_c": num("temperature_c", 36.2, 38.0),
            "wbc_count": num("wbc_count", 3500, 11000),
            "crp_mg_l": num("crp_mg_l", 0, 25),
            "cough_duration_days": random.randint(0, 2),
        })
    v.update(categorical_test_results(label))
    return v

def display_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)

def create_pdf(report_id, patient_id, report_date, layout, values, label):
    path = PDF_DIR / f"{report_id}.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        rightMargin=10*mm, leftMargin=10*mm,
        topMargin=10*mm, bottomMargin=10*mm
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "SyntheticTitle", parent=styles["Title"],
        fontSize=14 if layout == "compact" else 16,
        alignment=TA_CENTER, spaceAfter=5
    )
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7, leading=9)
    story = [
        Paragraph("SYNTHETIC INFECTIOUS-DISEASE LAB REPORT", title),
        Paragraph("FICTIONAL DATA — NOT FOR CLINICAL USE", styles["Heading3"]),
        Spacer(1, 5),
    ]
    meta = Table([
        ["Synthetic Patient ID", patient_id],
        ["Report ID", report_id],
        ["Report Date", report_date.isoformat()],
        ["Report Type", "Synthetic communicable-disease research sample"],
    ], colWidths=[42*mm, 90*mm])
    meta.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([meta, Spacer(1, 7)])
    rows = [["Feature", "Result", "Unit / possible values"]]
    units = {
        "fever_duration_days": "days", "temperature_c": "°C",
        "wbc_count": "cells/uL", "platelet_count": "cells/uL",
        "hemoglobin": "g/dL", "neutrophil_percent": "%",
        "lymphocyte_percent": "%", "crp_mg_l": "mg/L",
        "procalcitonin_ng_ml": "ng/mL", "oxygen_saturation_percent": "%",
        "cough_duration_days": "days",
    }
    for key in NUMERIC:
        rows.append([key, display_value(values[key]), units[key]])
    for key in TEST_FEATURES:
        rows.append([TEST_NAMES[key], values[key], "POSITIVE / NEGATIVE / NOT_PERFORMED / INDETERMINATE"])
    table = Table(rows, colWidths=[55*mm, 35*mm, 90*mm], repeatRows=1)
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7 if layout == "compact" else 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
    ]
    if layout == "classic":
        commands.extend([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ])
    elif layout == "compact":
        commands.append(("LINEBELOW", (0, 1), (-1, -1), 0.2, colors.lightgrey))
    table.setStyle(TableStyle(commands))
    story.extend([table, Spacer(1, 6)])
    story.append(Paragraph(
        "Simulation note: all values, results and category labels are generated "
        "for software research. This document is not a real medical report. "
        "The assigned category is a synthetic target label, not a diagnosis.",
        small
    ))
    doc.build(story)

def main():
    csv_path = LABEL_DIR / f"ground_truth_{REPORT_COUNT}.csv"
    fieldnames = [
        "report_id", "synthetic_patient_id", "report_date",
        "layout", *FEATURES, TARGET
    ]
    start_date = date(2025, 1, 1)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, REPORT_COUNT + 1):
            report_id = f"DEMO-{i:05d}"
            patient_id = f"SYN-{i:05d}"
            label = CLASSES[(i - 1) % len(CLASSES)]
            layout = LAYOUTS[(i - 1) % len(LAYOUTS)]
            report_date = start_date + timedelta(days=random.randint(0, 700))
            values = generate_features(label)
            create_pdf(report_id, patient_id, report_date, layout, values, label)
            row = {
                "report_id": report_id,
                "synthetic_patient_id": patient_id,
                "report_date": report_date.isoformat(),
                "layout": layout,
                **values,
                TARGET: label,
            }
            writer.writerow(row)
            if i % 100 == 0:
                print(f"Generated {i}/{REPORT_COUNT} reports")
    print("\nGENERATION COMPLETE")
    print(f"PDFs: {PDF_DIR}")
    print(f"CSV: {csv_path}")
    print(f"Reports: {REPORT_COUNT}; numeric features: {len(NUMERIC)}; test-result features: {len(TEST_FEATURES)}")
    print("Reminder: synthetic classes are not real diagnoses.")

if __name__ == "__main__":
    main()

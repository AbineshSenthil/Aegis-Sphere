import json, random

# --- Base case templates ---
# Each pair = (AI's wrong/suboptimal output, clinician's correction)

BASE_CASES = [
    {
        "patient_profile": {"age_band": "35-45", "hiv_status": True, "cancer_type": "B-cell_lymphoma", "comorbidities": ["HIV+", "pulmonary_TB_suspect"]},
        "ai_staging": "IIB", "ai_regimen": "R-CHOP",
        "correct_staging": "IVB", "correct_regimen": "CHOP + Liposomal_Doxorubicin",
        "reason": "Bone marrow involvement on biopsy confirms Stage IVB. Rituximab unavailable locally."
    },
    {
        "patient_profile": {"age_band": "25-35", "hiv_status": True, "cancer_type": "Kaposi_sarcoma", "comorbidities": ["HIV+"]},
        "ai_staging": "T1 I0 S0", "ai_regimen": "Liposomal_Doxorubicin",
        "correct_staging": "T1 I1 S1", "correct_regimen": "Liposomal_Doxorubicin + ART optimization",
        "reason": "Immune suppression (CD4=85) and systemic symptoms upgrade to S1. ART regimen needs review."
    },
    {
        "patient_profile": {"age_band": "45-55", "hiv_status": True, "cancer_type": "B-cell_lymphoma", "comorbidities": ["HIV+", "renal_impairment"]},
        "ai_staging": "IIA", "ai_regimen": "CHOP + Methotrexate",
        "correct_staging": "IIA", "correct_regimen": "CHOP (Methotrexate contraindicated — eGFR < 30)",
        "reason": "Renal impairment makes Methotrexate unsafe. Standard CHOP without MTX."
    },
    {
        "patient_profile": {"age_band": "30-40", "hiv_status": True, "cancer_type": "lung_adenocarcinoma", "comorbidities": ["HIV+", "active_TB"]},
        "ai_staging": "IIIA", "ai_regimen": "Carboplatin + Paclitaxel",
        "correct_staging": "IIIA", "correct_regimen": "Defer chemo — treat TB first. Reassess in 8 weeks.",
        "reason": "Active TB must be treated before cytotoxic chemotherapy. AI missed active infection flag."
    },
    {
        "patient_profile": {"age_band": "50-60", "hiv_status": False, "cancer_type": "cervical_cancer", "comorbidities": ["diabetes"]},
        "ai_staging": "IIB", "ai_regimen": "Cisplatin + RT",
        "correct_staging": "IIB", "correct_regimen": "Carboplatin + RT (Cisplatin out of stock)",
        "reason": "Cisplatin unavailable. Carboplatin is equivalent in concurrent chemoradiation for cervical cancer."
    },
]

# Generate 25 pairs by augmenting the 5 base cases with small variations
def generate_pairs(base_cases, n_total=25):
    pairs = []
    age_bands = ["25-35", "35-45", "45-55", "55-65"]
    cd4_levels = ["CD4=45", "CD4=120", "CD4=85", "CD4=200", "CD4=310"]

    for i in range(n_total):
        base = base_cases[i % len(base_cases)].copy()
        profile = base["patient_profile"].copy()
        profile["age_band"] = random.choice(age_bands)
        profile["cd4_note"] = random.choice(cd4_levels)
        base["patient_profile"] = profile
        base["pair_id"] = f"PAIR_{i+1:03d}"

        instruction = (
            "You are an oncology AI assistant for an LMIC TB/HIV clinic. "
            "Given the patient case below, output the correct TNM staging and "
            "a resource-constrained treatment regimen that respects local drug availability."
        )

        input_text = (
            f"Patient profile: {json.dumps(profile)}\n"
            f"AI proposed staging: {base['ai_staging']}\n"
            f"AI proposed regimen: {base['ai_regimen']}\n"
            f"Local inventory note: drugs may be out of stock."
        )

        output_text = (
            f"Corrected staging: {base['correct_staging']}\n"
            f"Corrected regimen: {base['correct_regimen']}\n"
            f"Clinical reason: {base['reason']}"
        )

        pairs.append({
            "pair_id": base["pair_id"],
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "ai_was_wrong_about": "staging" if base["ai_staging"] != base["correct_staging"] else "regimen",
                "patient_profile": profile
            }
        })

    return pairs

pairs = generate_pairs(BASE_CASES, n_total=25)

with open("data/lora_training_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)

print(f"Generated {len(pairs)} LoRA training pairs")
print("Sample pair:")
print(json.dumps(pairs[0], indent=2))

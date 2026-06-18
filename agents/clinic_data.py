"""
Shared clinic-data access for the Astralace agents.

Pure data layer: JSON loaders and the specialty -> provider_type mapping. No LLM
imports here, so both the care-coordination agent and the care-matching agent can
import it without pulling in langchain, and the deterministic logic stays testable
without an API key.
"""

import json
from pathlib import Path

MOCK_DATA_DIR = Path(__file__).parent.parent / "mock-data"


def load_json(filename: str) -> dict:
    with open(MOCK_DATA_DIR / filename) as f:
        return json.load(f)


def get_patient(patient_id: str) -> dict:
    data = load_json("patients.json")
    for p in data["patients"]:
        if p["id"] == patient_id:
            return p
    valid = ", ".join(p["id"] for p in data["patients"][:10])
    raise ValueError(f"Patient '{patient_id}' not found. First available IDs: {valid} ...")


def get_providers() -> list[dict]:
    return load_json("providers.json")["providers"]


def get_pathway(pathway_id: str) -> dict:
    data = load_json("care-pathways.json")
    for p in data["pathways"]:
        if p["id"] == pathway_id:
            return p
    raise ValueError(f"Pathway {pathway_id} not found")


# Specialty-string -> pathway provider_type. Single source of truth shared by
# every agent's filter so the mapping can never drift between them.
_SPECIALTY_MAP = [
    ("obstetrics", "ob_gyn"),
    ("prenatal", "ob_gyn"),
    ("lactation", "lactation_consultant"),
    ("infant feeding", "lactation_consultant"),
    ("pelvic floor", "pelvic_floor_pt"),
    ("nutrition", "registered_dietitian"),
    ("dietitian", "registered_dietitian"),
    ("mental health", "therapist"),
    ("therapy", "therapist"),
    ("therapist", "therapist"),
    ("lcsw", "therapist"),
    ("lpc", "therapist"),
    ("psychiatry", "psychiatrist"),
    ("pediatrics", "pediatrician"),
    ("midwife", "midwife"),
    ("career", "career_coach"),
    ("return-to-work", "career_coach"),
    ("maternity leave", "career_coach"),
    ("health coach", "health_coach"),
    ("fertility", "fertility_specialist"),
    ("reproductive endocrin", "fertility_specialist"),
    ("menopause", "menopause_specialist"),
]


def infer_provider_type(prov: dict) -> str:
    """Map a provider's specialty string to the provider_type keys used in
    care-pathways.json."""
    specialty = prov.get("specialty", "").lower()
    for keyword, ptype in _SPECIALTY_MAP:
        if keyword in specialty:
            return ptype
    return "general"

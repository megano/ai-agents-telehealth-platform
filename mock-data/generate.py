#!/usr/bin/env python3
"""
Synthetic data generator for Astralace Women's Health (portfolio demo).

Single source of truth for mock-data/providers.json and mock-data/patients.json.
Seeded and reproducible: the same --seed always yields the same dataset, so the
demo and eval cases are pinned by seed rather than hand-authored. All data is
synthetic; identifiers follow RFC-reserved conventions (example.com, 555-01xx).

Two layers, both produced by the same builders so the schema is uniform:
  1. CURATED records (fixed): the three anchor patients that schedules.json /
     comms-queue.json reference and that the care-coordination agent demos
     (pat_002/003/004), plus a handful of scenario fixtures that exercise the
     matcher's interesting paths (0-results relaxation, language hard constraint,
     implicit-signal re-rank). These are deterministic, not random.
  2. BULK records (seeded random): a generous, realistically-distributed set so
     the matcher's eligibility filter + weighted ranking have a real pool to work
     over and eval can measure behavior at volume.

Usage:
    python mock-data/generate.py                       # defaults: 80 providers, 300 patients
    python mock-data/generate.py --providers 150 --patients 500
    python mock-data/generate.py --seed 7 --out mock-data

Scope: maternity / perinatal journey only (prenatal -> postpartum). No fertility,
pediatrics-as-own-patient, or menopause patients (cut by design decision 2026-06-16).
"""

import argparse
import json
import random
from pathlib import Path

# ── Reference pools ─────────────────────────────────────────────────────────

# State -> (representative city, IANA timezone). Kept to states where the clinic
# realistically has license coverage so generated records stay internally valid.
STATES = {
    "TX": ("Austin", "America/Chicago"),
    "CA": ("Los Angeles", "America/Los_Angeles"),
    "NY": ("New York", "America/New_York"),
    "FL": ("Miami", "America/New_York"),
    "CO": ("Denver", "America/Denver"),
    "WA": ("Seattle", "America/Los_Angeles"),
    "NM": ("Santa Fe", "America/Denver"),
    "OR": ("Portland", "America/Los_Angeles"),
    "NJ": ("Newark", "America/New_York"),
    "CT": ("Hartford", "America/New_York"),
    "NV": ("Las Vegas", "America/Los_Angeles"),
    "AZ": ("Phoenix", "America/Phoenix"),
    "GA": ("Atlanta", "America/New_York"),
    "IL": ("Chicago", "America/Chicago"),
    "MA": ("Boston", "America/New_York"),
}
STATE_LIST = list(STATES)

HUBS = {"TX": "hub_atx", "NY": "hub_nyc", "CA": "hub_la"}

LANGUAGES = ["English", "Spanish", "Mandarin", "Hindi"]
INSURERS = ["Aetna", "Cigna", "Blue Cross Blue Shield", "UnitedHealth", "Humana", "Self-pay"]

FIRST_F = ["Sofia", "Denise", "Rebecca", "Amara", "Priya", "Maya", "Elena", "Nadia",
           "Grace", "Leah", "Carmen", "Ingrid", "Hana", "Tara", "Yuki", "Fatima",
           "Olivia", "Camila", "Aisha", "Wren", "Naomi", "Daniela", "Simone", "Ruth"]
FIRST_M = ["Marcus", "David", "Andre", "Ben", "Theo", "Omar", "Liam", "Noah", "Caleb"]
LAST = ["Nguyen", "Park", "Hill", "Johnson", "Sharma", "Lopez", "Okafor", "Reyes",
        "Kim", "Patel", "Brooks", "Webb", "Morrison", "Adeyemi", "Torres", "Chen",
        "Gonzalez", "Hammond", "Moore", "Vasquez", "Mbeki", "Larsson", "Tanaka"]

# Therapy modalities, only meaningful for therapist-type providers.
THERAPY_MODALITIES = ["CBT", "EMDR", "IFS", "DBT", "ACT", "psychodynamic"]

DAYPARTS = ["morning", "afternoon", "evening"]
DAYPART_HOURS = {"morning": "8-12", "afternoon": "12-17", "evening": "17-20"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# provider_type -> generation profile. The `specialty` string MUST contain a
# keyword that care_coordination_agent._infer_provider_type maps back to this
# type, or the agent's provider filter will silently drop the provider.
ROLES = {
    "ob_gyn": {
        "weight": 14, "specialty": "Obstetrics & Prenatal Care",
        "creds": ["MD", "MD, MPH", "DO"], "gender_p_female": 0.7,
        "requires_license": True, "is_therapist": False,
    },
    "therapist": {
        "weight": 16, "specialty": "Perinatal Mental Health & Therapy",
        "creds": ["LCSW, PMH-C", "LPC, NCC", "PsyD", "LMFT"], "gender_p_female": 0.8,
        "requires_license": True, "is_therapist": True,
    },
    "psychiatrist": {
        "weight": 5, "specialty": "Psychiatry (Perinatal)",
        "creds": ["MD, ABPN"], "gender_p_female": 0.6,
        "requires_license": True, "is_therapist": False,
    },
    "lactation_consultant": {
        "weight": 10, "specialty": "Lactation & Infant Feeding",
        "creds": ["RN, IBCLC", "IBCLC"], "gender_p_female": 0.95,
        "requires_license": True, "is_therapist": False,
    },
    "registered_dietitian": {
        "weight": 9, "specialty": "Prenatal & Postpartum Nutrition (Dietitian)",
        "creds": ["RD, CLEC", "RDN"], "gender_p_female": 0.85,
        "requires_license": True, "is_therapist": False,
    },
    "pelvic_floor_pt": {
        "weight": 8, "specialty": "Pelvic Floor Physical Therapy",
        "creds": ["PT, DPT, WCS", "PT, DPT"], "gender_p_female": 0.8,
        "requires_license": True, "is_therapist": False,
    },
    "midwife": {
        "weight": 6, "specialty": "Certified Nurse-Midwife",
        "creds": ["CNM, MSN"], "gender_p_female": 0.95,
        "requires_license": True, "is_therapist": False,
    },
    # Pediatrician staffs the T3 "prenatal pediatrician intro" touchpoint for the
    # mother's care team. Pediatric *patients* are out of scope, but this team
    # role stays in the maternity pathway (decision 2026-06-16).
    "pediatrician": {
        "weight": 5, "specialty": "Pediatrics & Newborn Care",
        "creds": ["MD, FAAP"], "gender_p_female": 0.6,
        "requires_license": True, "is_therapist": False,
    },
    "career_coach": {
        "weight": 4, "specialty": "Maternity Leave & Return-to-Work Coaching",
        "creds": ["CPC, CPCC", "ACC"], "gender_p_female": 0.7,
        "requires_license": False, "is_therapist": False,
    },
    "health_coach": {
        "weight": 4, "specialty": "Perinatal Health Coach",
        "creds": ["MS, CHC", "NBC-HWC"], "gender_p_female": 0.6,
        "requires_license": False, "is_therapist": False,
    },
}
ROLE_LIST = list(ROLES)
ROLE_WEIGHTS = [ROLES[r]["weight"] for r in ROLE_LIST]

# Which roles a patient at a given care stage is actively matched against. Used
# to give bulk patients sensible assigned providers and chief complaints.
STAGE_PRIMARY_ROLE = {
    "T1": "ob_gyn", "T2": "ob_gyn", "T3": "ob_gyn",
    "postpartum": "therapist",
}
STAGE_PRODUCT_AREA = {
    "T1": "Pregnancy & Prenatal Care", "T2": "Pregnancy & Prenatal Care",
    "T3": "Pregnancy & Prenatal Care", "postpartum": "Postpartum & Fourth Trimester",
}

PERINATAL_PRODUCT_AREAS = ["Pregnancy & Prenatal Care", "Postpartum & Fourth Trimester",
                           "Mental & Behavioral Health"]

# Roles the maternity pathway requires across all stages (license-exempt coach
# roles are "all"-licensed and always covered, so they are omitted here). The
# demo states are guaranteed full coverage of these so the anchor/scenario
# patients staff cleanly.
PATHWAY_ROLES = {"ob_gyn", "registered_dietitian", "therapist", "pelvic_floor_pt",
                 "lactation_consultant", "pediatrician"}
DEMO_STATES = ["TX", "CA", "NY", "NM", "CO"]


# ── Builders ────────────────────────────────────────────────────────────────

def _phone(rng):
    """RFC-reserved fictional number: <area>-555-01XX."""
    area = rng.choice([212, 213, 312, 415, 512, 646, 718, 917, 305, 303, 206])
    return f"+1{area}555{rng.randint(100, 199):04d}"


def _gender(rng, p_female):
    return "female" if rng.random() < p_female else "male"


def _dayparts(rng):
    """1-3 dayparts a provider works; biased toward daytime, evening rarer."""
    picks = []
    if rng.random() < 0.85:
        picks.append("morning")
    if rng.random() < 0.85:
        picks.append("afternoon")
    if rng.random() < 0.35:
        picks.append("evening")
    return picks or ["afternoon"]


def _weekly_slots(rng, dayparts):
    """Build weekly_slots consistent with the chosen dayparts."""
    days = rng.sample(WEEKDAYS, k=rng.randint(3, 5))
    hours = "-".join([DAYPART_HOURS[dayparts[0]].split("-")[0],
                      DAYPART_HOURS[dayparts[-1]].split("-")[1]])
    return [f"{d} {hours}" for d in sorted(days, key=WEEKDAYS.index)]


def _licensed_states(rng, requires_license):
    if not requires_license:
        return "all"
    return rng.sample(STATE_LIST, k=rng.randint(2, 5))


def _languages(rng):
    langs = ["English"]
    if rng.random() < 0.30:
        langs.append(rng.choice(["Spanish", "Mandarin", "Hindi"]))
    return langs


def make_provider(rng, pid, role):
    prof = ROLES[role]
    gender = _gender(rng, prof["gender_p_female"])
    name = f"{rng.choice(FIRST_F if gender == 'female' else FIRST_M)} {rng.choice(LAST)}"
    cred = rng.choice(prof["creds"])
    # Add a credential suffix to the display name, matching the existing style.
    display = f"Dr. {name}" if cred.startswith(("MD", "DO", "PsyD", "PT")) else f"{name}, {cred.split(',')[0]}"
    licensed = _licensed_states(rng, prof["requires_license"])
    dayparts = _dayparts(rng)
    state_for_hub = (licensed[0] if isinstance(licensed, list) else rng.choice(STATE_LIST))

    prov = {
        "id": pid,
        "name": display,
        "credentials": cred,
        "specialty": prof["specialty"],
        "product_areas": [STAGE_PRODUCT_AREA["T2"], STAGE_PRODUCT_AREA["postpartum"]],
        "gender": gender,
        "hub": HUBS.get(state_for_hub),
        "virtual_available": True,
        "licensed_states": licensed,
        "languages": _languages(rng),
        "availability_dayparts": dayparts,
        "availability": {
            "timezone": STATES.get(state_for_hub, ("", "America/Chicago"))[1],
            "weekly_slots": _weekly_slots(rng, dayparts),
        },
        "avg_rating": round(rng.uniform(4.3, 5.0), 1),
        "total_reviews": rng.randint(40, 600),
        "insurance": sorted(rng.sample(INSURERS, k=rng.randint(2, 4))),
        # Most accept new patients; a minority are closed to exercise the filter.
        "accepting_new_patients": rng.random() < 0.85,
    }
    if not prof["requires_license"]:
        prov["state_license_required"] = False
    if prof["is_therapist"]:
        prov["modalities"] = sorted(rng.sample(THERAPY_MODALITIES, k=rng.randint(2, 4)))
    return prov


def _pref(category, preference, applies_to, source, strength, tier, weight, provenance):
    """One preference_profile entry (the finalized 8-field schema)."""
    return {
        "category": category, "preference": preference, "applies_to": applies_to,
        "source": source, "strength": strength, "tier": tier,
        "weight": weight, "provenance": provenance,
    }


def make_patient(rng, pid, stage=None):
    stage = stage or rng.choices(["T1", "T2", "T3", "postpartum"], weights=[2, 3, 3, 4])[0]
    gender_pref = rng.choices(["female", "no_preference"], weights=[1, 1])[0]
    modality = rng.choices(["virtual", "in-person", "no_preference"], weights=[5, 2, 2])[0]
    language = rng.choices(LANGUAGES, weights=[80, 10, 6, 4])[0]
    state = rng.choice(STATE_LIST)
    city = STATES[state][0]
    daypart = rng.choice(DAYPARTS)
    continuity = rng.random() < 0.5
    first = rng.choice(FIRST_F)
    name = f"{first} {rng.choice(LAST)}"
    birth_year = rng.randint(1985, 2000)

    # Build a weighted preference_profile from the same signals as the legacy
    # matching_preferences, so the two views agree. Explicit asks weigh highest.
    profile = []
    if gender_pref == "female":
        profile.append(_pref("gender", "female", "therapist", "intake", "explicit",
                             "soft", 0.7, "Intake form: provider gender preference"))
    if language != "English":
        profile.append(_pref("language", language, "all", "intake", "explicit",
                             "hard_constraint", 1.0, "Intake form: primary language"))
    profile.append(_pref("scheduling", daypart, "all", "intake", "explicit",
                         "soft" if rng.random() < 0.7 else "hard_constraint",
                         round(rng.uniform(0.5, 0.9), 2), "Intake form: preferred time of day"))
    if modality != "no_preference":
        profile.append(_pref("modality", modality, "all", "intake", "explicit",
                             "soft", round(rng.uniform(0.4, 0.7), 2),
                             "Intake form: visit modality preference"))
    if continuity:
        profile.append(_pref("continuity", "retain_current_provider", "all", "behavioral",
                             "inferred", "soft", 0.4,
                             "Scheduling pattern: rebooks with same provider"))

    return {
        "id": pid,
        "name": name,
        "dob": f"{birth_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "age": 2026 - birth_year,
        "location": {"city": city, "state": state},
        "language": language,
        "insurance": rng.choice(INSURERS),
        "product_area": STAGE_PRODUCT_AREA[stage],
        "care_status": "active",
        "care_stage": stage,
        **({"pregnancy_week": {"T1": rng.randint(4, 12), "T2": rng.randint(13, 27),
                               "T3": rng.randint(28, 40)}[stage]} if stage != "postpartum"
           else {"postpartum_week": rng.randint(1, 12)}),
        "assigned_provider": None,  # filled in after providers exist
        "chief_complaint": _chief_complaint(stage),
        "matching_preferences": {
            "provider_gender": gender_pref,
            "language": language,
            "modality": modality if modality != "no_preference" else "no_preference",
            "continuity": continuity,
            "notes": "Generated synthetic patient.",
        },
        "intake_note": _intake_note(first, stage, gender_pref, modality, daypart, language),
        "signals": [],
        "preference_profile": profile,
        "contact": {
            "phone": _phone(rng),
            "email": f"{first.lower()}.{name.split()[1][0].lower()}@example.com",
            "sms_opt_in": rng.random() < 0.8,
        },
    }


def _chief_complaint(stage):
    return {
        "T1": "First trimester prenatal care; establishing care team",
        "T2": "Second trimester prenatal monitoring",
        "T3": "Third trimester; preparing postpartum care team",
        "postpartum": "Postpartum recovery and mental health support",
    }[stage]


def _intake_note(first, stage, gender_pref, modality, daypart, language):
    bits = [f"{first} is in the {stage} stage of perinatal care."]
    if gender_pref == "female":
        bits.append("Prefers a female provider for therapy.")
    if modality != "no_preference":
        bits.append(f"Prefers {modality} visits.")
    bits.append(f"Best availability is in the {daypart}.")
    if language != "English":
        bits.append(f"Primary language is {language}; requires a language-matched provider.")
    return " ".join(bits)


def assign_providers(rng, patients, providers):
    """Give each patient an assigned_provider that is actually plausible:
    a provider of the stage's primary role, licensed in the patient's state."""
    by_role_state = {}
    for p in providers:
        role = _role_of(p)
        lic = p["licensed_states"]
        states = STATE_LIST if lic == "all" else lic
        for s in states:
            by_role_state.setdefault((role, s), []).append(p["id"])

    for pat in patients:
        if pat["assigned_provider"] is not None:
            continue  # curated patients keep their fixed assignment
        role = STAGE_PRIMARY_ROLE[pat["care_stage"]]
        pool = by_role_state.get((role, pat["location"]["state"]))
        pat["assigned_provider"] = rng.choice(pool) if pool else None


def _role_of(prov):
    """Recover provider_type from specialty, mirroring the agent's mapping."""
    s = prov["specialty"].lower()
    table = [("obstetrics", "ob_gyn"), ("prenatal", "ob_gyn"),
             ("lactation", "lactation_consultant"), ("pelvic floor", "pelvic_floor_pt"),
             ("nutrition", "registered_dietitian"), ("dietitian", "registered_dietitian"),
             ("psychiatry", "psychiatrist"), ("mental health", "therapist"),
             ("therapy", "therapist"), ("midwife", "midwife"),
             ("return-to-work", "career_coach"), ("maternity leave", "career_coach"),
             ("health coach", "health_coach")]
    for kw, role in table:
        if kw in s:
            return role
    return "general"


# ── Curated records (deterministic; not random) ──────────────────────────────
# These pin the demo. The three anchors keep schedules.json / comms-queue.json
# valid and keep the care-coordination agent's default demo (pat_002) working;
# they are the prior hand-written records, now carrying the new matcher schema.
# The scenario fixtures exercise the matcher's interesting paths and are selected
# by their fixed IDs in the demo and eval.

def curated_providers():
    """The named perinatal team from the original mock data, enriched with the
    new matcher fields (gender, modalities, availability_dayparts)."""
    return [
        _cp("prov_002", "Dr. Maria Gonzalez", "MD, MPH", "Obstetrics & Prenatal Care",
            "female", ["TX", "CO", "FL", "NM"], ["English", "Spanish"],
            ["morning", "afternoon"], "America/Chicago",
            ["Mon 8-16", "Wed 8-16", "Thu 8-16"], 4.8, 198,
            ["Aetna", "UnitedHealth", "Humana", "Self-pay"]),
        _cp("prov_003", "Dr. Janet Kim", "PsyD, PMH-C", "Perinatal Mental Health & Therapy",
            "female", ["CA", "WA", "OR", "NV"], ["English", "Mandarin"],
            ["afternoon", "evening"], "America/Los_Angeles",
            ["Tue 10-18", "Wed 10-18", "Fri 10-15"], 4.9, 441,
            ["Cigna", "Blue Cross Blue Shield", "Self-pay"],
            modalities=["CBT", "EMDR", "IFS"]),
        _cp("prov_004", "Lauren Chen, IBCLC", "RN, IBCLC", "Lactation & Infant Feeding",
            "female", ["NY", "NJ", "CT", "TX"], ["English", "Mandarin"],
            ["morning", "afternoon"], "America/New_York",
            ["Mon 10-17", "Tue 10-17", "Thu 10-17", "Sat 9-13"], 4.7, 276,
            ["Aetna", "Cigna", "Self-pay"]),
    ]


def _cp(pid, name, cred, specialty, gender, states, langs, dayparts, tz, slots,
        rating, reviews, insurance, modalities=None):
    prov = {
        "id": pid, "name": name, "credentials": cred, "specialty": specialty,
        "product_areas": ["Pregnancy & Prenatal Care", "Postpartum & Fourth Trimester"],
        "gender": gender, "hub": HUBS.get(states[0]), "virtual_available": True,
        "licensed_states": states, "languages": langs,
        "availability_dayparts": dayparts,
        "availability": {"timezone": tz, "weekly_slots": slots},
        "avg_rating": rating, "total_reviews": reviews,
        "insurance": insurance, "accepting_new_patients": True,
    }
    if modalities:
        prov["modalities"] = modalities
    return prov


def curated_patients():
    """Three anchors (enriched) + scenario fixtures for the matcher demo."""
    pats = []

    # Anchor 1, Sofia (pat_002): prenatal, in-person + continuity. Default demo
    # patient for the care-coordination agent. Per-role team match scenario.
    pats.append({
        "id": "pat_002", "name": "Sofia Nguyen", "dob": "1994-11-02", "age": 29,
        "location": {"city": "Austin", "state": "TX"}, "language": "English",
        "insurance": "UnitedHealth", "product_area": "Pregnancy & Prenatal Care",
        "care_status": "active", "care_stage": "T2", "pregnancy_week": 26,
        "assigned_provider": "prov_002",
        "chief_complaint": "Prenatal care, 26 weeks pregnant, first pregnancy",
        "visit_history": [
            {"date": "2026-02-01", "type": "OB Intake", "provider": "prov_002",
             "notes": "EDD Aug 12. NT scan normal. NIPT ordered."},
            {"date": "2026-03-10", "type": "Prenatal Check-in (14 wk)", "provider": "prov_002",
             "notes": "BP 118/72. FHR 152. Anatomy scan scheduled at 20 wk."},
        ],
        "next_appointment": "2026-05-21T09:00:00-06:00",
        "matching_preferences": {
            "provider_gender": "no_preference", "language": "English",
            "modality": "in-person", "continuity": True,
            "notes": "Prefers in-person for pregnancy visits; continuity with Dr. Gonzalez important",
        },
        "intake_note": "Sofia is 26 weeks into her first pregnancy. Prefers in-person visits "
                       "for her pregnancy care and wants to keep seeing Dr. Gonzalez. Daytime "
                       "availability around work.",
        "signals": [],
        "preference_profile": [
            _pref("modality", "in-person", "ob_gyn", "intake", "explicit", "soft", 0.7,
                  "Intake form: prefers in-person pregnancy visits"),
            _pref("continuity", "prov_002", "ob_gyn", "intake", "explicit", "soft", 0.8,
                  "Intake form: wants to keep current OB"),
            _pref("scheduling", "afternoon", "all", "behavioral", "inferred", "soft", 0.4,
                  "Scheduling pattern: books afternoon slots"),
        ],
        "contact": {"phone": "+15125550348", "email": "sofia.n@example.com", "sms_opt_in": True},
    })

    # Anchor 2, Denise (pat_003): postpartum PPD. Flagship weighted-match case:
    # female + virtual + evening + CBT, multiple soft prefs the matcher ranks on.
    pats.append({
        "id": "pat_003", "name": "Denise Park", "dob": "1988-07-19", "age": 37,
        "location": {"city": "Los Angeles", "state": "CA"}, "language": "English",
        "insurance": "Cigna", "product_area": "Postpartum & Fourth Trimester",
        "care_status": "active", "care_stage": "postpartum", "postpartum_week": 10,
        "assigned_provider": "prov_003",
        "chief_complaint": "Postpartum depression symptoms, 10 weeks postpartum",
        "visit_history": [
            {"date": "2026-03-28", "type": "Postpartum Mental Health Intake", "provider": "prov_003",
             "notes": "Edinburgh score 14. Mild-moderate PPD. Weekly therapy recommended."},
            {"date": "2026-04-11", "type": "Therapy Session", "provider": "prov_003",
             "notes": "CBT session. Sleep hygiene, support network mapping."},
        ],
        "next_appointment": "2026-05-16T11:00:00-08:00",
        "matching_preferences": {
            "provider_gender": "female", "language": "English", "modality": "virtual",
            "continuity": True,
            "notes": "Strong preference for female therapist; feels more comfortable discussing birth trauma",
        },
        "intake_note": "Denise is 10 weeks postpartum with mild-moderate PPD. Strongly prefers a "
                       "female therapist, virtual sessions, and evenings after the baby is down. "
                       "Has responded well to CBT.",
        "signals": [],
        "preference_profile": [
            _pref("gender", "female", "therapist", "intake", "explicit", "soft", 0.9,
                  "Intake form: strong preference for female therapist"),
            _pref("modality", "virtual", "therapist", "intake", "explicit", "soft", 0.7,
                  "Intake form: prefers virtual therapy"),
            _pref("scheduling", "evening", "therapist", "intake", "explicit", "soft", 0.6,
                  "Intake form: evenings after baby's bedtime"),
            _pref("modality_method", "CBT", "therapist", "behavioral", "inferred", "soft", 0.5,
                  "Visit history: responded well to CBT"),
        ],
        "contact": {"phone": "+13235550671", "email": "denise.p@example.com", "sms_opt_in": True},
    })

    # Anchor 3, Rebecca (pat_004): lactation, availability-first, no continuity.
    pats.append({
        "id": "pat_004", "name": "Rebecca Hill", "dob": "1991-04-30", "age": 33,
        "location": {"city": "New York", "state": "NY"}, "language": "English",
        "insurance": "Aetna", "product_area": "Postpartum & Fourth Trimester",
        "care_status": "active", "assigned_provider": "prov_004",
        "care_stage": "postpartum", "postpartum_week": 3,
        "chief_complaint": "Breastfeeding difficulty, low supply, latch issues",
        "visit_history": [
            {"date": "2026-04-05", "type": "Lactation Consult", "provider": "prov_004",
             "notes": "Latch assessment done. Nipple shield trial. Pumping schedule set."},
        ],
        "next_appointment": "2026-05-19T14:00:00-05:00",
        "matching_preferences": {
            "provider_gender": "no_preference", "language": "English", "modality": "virtual",
            "continuity": False, "notes": "Open to any lactation specialist; availability is top priority",
        },
        "intake_note": "Rebecca is 3 weeks postpartum with latch and supply issues. No provider "
                       "preference; wants the soonest available lactation appointment. Mornings work best.",
        "signals": [],
        "preference_profile": [
            _pref("scheduling", "morning", "lactation_consultant", "intake", "explicit", "soft", 0.8,
                  "Intake form: mornings preferred, soonest availability is priority"),
            _pref("modality", "virtual", "lactation_consultant", "intake", "explicit", "soft", 0.5,
                  "Intake form: open to virtual"),
        ],
        "contact": {"phone": "+12125550924", "email": "rebecca.h@example.com", "sms_opt_in": False},
    })

    # Scenario A: language hard constraint (Spanish, prenatal, TX). Filter must
    # keep only Spanish-speaking OBs licensed in TX.
    pats.append({
        "id": "pat_020", "name": "Carmen Lopez", "dob": "1996-02-12", "age": 30,
        "location": {"city": "Houston", "state": "TX"}, "language": "Spanish",
        "insurance": "Aetna", "product_area": "Pregnancy & Prenatal Care",
        "care_status": "active", "care_stage": "T1", "pregnancy_week": 9,
        "assigned_provider": None,
        "chief_complaint": "Early prenatal care, Spanish-speaking, first pregnancy",
        "next_appointment": None,
        "matching_preferences": {
            "provider_gender": "female", "language": "Spanish", "modality": "virtual",
            "continuity": False, "notes": "Requires Spanish-speaking provider",
        },
        "intake_note": "Carmen is 9 weeks pregnant and speaks primarily Spanish. Requires a "
                       "Spanish-speaking provider. Prefers a female OB and virtual visits.",
        "signals": [],
        "preference_profile": [
            _pref("language", "Spanish", "all", "intake", "explicit", "hard_constraint", 1.0,
                  "Intake form: requires Spanish-speaking provider"),
            _pref("gender", "female", "ob_gyn", "intake", "explicit", "soft", 0.7,
                  "Intake form: prefers female OB"),
            _pref("modality", "virtual", "all", "intake", "explicit", "soft", 0.5,
                  "Intake form: prefers virtual"),
        ],
        "contact": {"phone": "+17135550142", "email": "carmen.l@example.com", "sms_opt_in": True},
    })

    # Scenario B: 0-results -> relaxation. A firm combination almost no provider
    # meets: a Hindi-speaking, evening-only pelvic floor PT in NM. Expect the
    # eligibility filter to return empty and the relaxation pass to fire.
    pats.append({
        "id": "pat_021", "name": "Priya Sharma", "dob": "1992-09-03", "age": 33,
        "location": {"city": "Santa Fe", "state": "NM"}, "language": "Hindi",
        "insurance": "Cigna", "product_area": "Postpartum & Fourth Trimester",
        "care_status": "active", "care_stage": "postpartum", "postpartum_week": 7,
        "assigned_provider": None,
        "chief_complaint": "Postpartum pelvic floor recovery; needs Hindi-speaking PT, evenings only",
        "next_appointment": None,
        "matching_preferences": {
            "provider_gender": "female", "language": "Hindi", "modality": "virtual",
            "continuity": False, "notes": "Hindi-speaking; only available evenings",
        },
        "intake_note": "Priya is 7 weeks postpartum and needs pelvic floor PT. Speaks Hindi and "
                       "requires a language-matched provider. Only free in the evenings after work. "
                       "Lives in New Mexico, where network coverage is thin.",
        "signals": [],
        "preference_profile": [
            _pref("language", "Hindi", "all", "intake", "explicit", "hard_constraint", 1.0,
                  "Intake form: requires Hindi-speaking provider"),
            _pref("scheduling", "evening", "pelvic_floor_pt", "intake", "explicit",
                  "hard_constraint", 0.9, "Intake form: only available evenings"),
            _pref("gender", "female", "pelvic_floor_pt", "intake", "explicit", "soft", 0.6,
                  "Intake form: prefers female PT"),
        ],
        "contact": {"phone": "+15055550178", "email": "priya.s@example.com", "sms_opt_in": True},
    })

    # Scenario C: implicit-signal re-rank. Intake says no time preference, but a
    # later provider-change request reveals a hard before-3pm constraint (kid
    # pickup). The refinement beat: this signal spikes the scheduling weight.
    pats.append({
        "id": "pat_022", "name": "Maya Johnson", "dob": "1990-06-21", "age": 35,
        "location": {"city": "Denver", "state": "CO"}, "language": "English",
        "insurance": "UnitedHealth", "product_area": "Postpartum & Fourth Trimester",
        "care_status": "active", "care_stage": "postpartum", "postpartum_week": 9,
        "assigned_provider": None,
        "chief_complaint": "Postpartum therapy; scheduling friction surfaced via provider-change request",
        "next_appointment": None,
        "matching_preferences": {
            "provider_gender": "no_preference", "language": "English", "modality": "virtual",
            "continuity": False, "notes": "No stated time preference at intake",
        },
        "intake_note": "Maya is 9 weeks postpartum, seeking ongoing therapy. At intake she stated "
                       "no particular scheduling preference and is open to any therapist.",
        "signals": [
            {
                "id": "sig_022_1", "type": "provider_change_request",
                "received_at": "2026-05-12T16:40:00Z",
                "channel": "patient_message",
                "text": "I need to switch to someone who can do appointments before 3pm. "
                        "I have to leave for daycare pickup and keep missing my evening slots.",
                "derived_preference": {
                    "category": "scheduling", "preference": "afternoon",
                    "applies_to": "therapist", "tier": "hard_constraint",
                },
            },
        ],
        "preference_profile": [
            # Intake captured nothing firm; the strong signal is added by refinement.
            _pref("scheduling", "afternoon", "therapist", "request", "explicit",
                  "hard_constraint", 1.0,
                  "Provider-change request 2026-05-12: must finish before 3pm for daycare pickup"),
            _pref("modality", "virtual", "therapist", "intake", "explicit", "soft", 0.4,
                  "Intake form: open to virtual"),
        ],
        "contact": {"phone": "+13035550159", "email": "maya.j@example.com", "sms_opt_in": True},
    })

    return pats


# ── Assembly ──────────────────────────────────────────────────────────────────

def _covered(providers, role, state):
    """Is at least one provider of `role` licensed in `state`?"""
    for p in providers:
        if _role_of(p) != role:
            continue
        lic = p["licensed_states"]
        if lic == "all" or state in lic:
            return True
    return False


def build(seed, n_providers, n_patients):
    rng = random.Random(seed)

    providers = curated_providers()
    next_pid = 1000

    def add(role, force_state=None):
        nonlocal next_pid
        prov = make_provider(rng, f"prov_{next_pid}", role)
        # Guarantee the provider is licensed in a specific state when we are
        # filling a coverage gap (only meaningful for license-required roles).
        if force_state and isinstance(prov["licensed_states"], list) \
                and force_state not in prov["licensed_states"]:
            prov["licensed_states"][0] = force_state
        providers.append(prov)
        next_pid += 1

    # Coverage pass first, two guarantees:
    #  (a) Every state has the assignment roles (ob_gyn, therapist) so no active
    #      patient is left without an in-state provider.
    #  (b) The demo states fully staff the maternity pathway, so the anchor/
    #      scenario patients get complete care teams with no spurious unmet_needs.
    #      (Bulk patients in other states may hit real coverage gaps; that is
    #      realistic and the agent flags them via unmet_needs by design.)
    for role in sorted(set(STAGE_PRIMARY_ROLE.values())):
        for state in STATE_LIST:
            if not _covered(providers, role, state):
                add(role, force_state=state)
    for role in sorted(PATHWAY_ROLES):
        for state in DEMO_STATES:
            if not _covered(providers, role, state):
                add(role, force_state=state)

    # Then random fill to the requested total.
    while len(providers) < n_providers:
        add(rng.choices(ROLE_LIST, weights=ROLE_WEIGHTS)[0])

    patients = curated_patients()
    next_patid = 1000
    while len(patients) < n_patients:
        patients.append(make_patient(rng, f"pat_{next_patid}"))
        next_patid += 1

    assign_providers(rng, patients, providers)

    # Stable ordering for clean diffs: curated first (by id), then bulk.
    return (
        {"providers": providers},
        {"patients": patients},
    )


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic Astralace mock data.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42; pins the dataset)")
    ap.add_argument("--providers", type=int, default=80, help="total providers incl. curated (default 80)")
    ap.add_argument("--patients", type=int, default=300, help="total patients incl. curated (default 300)")
    ap.add_argument("--out", default=str(Path(__file__).parent), help="output directory")
    args = ap.parse_args()

    if args.providers < 3 or args.patients < 6:
        raise SystemExit("Need room for the curated records: --providers >= 3, --patients >= 6")

    providers, patients = build(args.seed, args.providers, args.patients)
    out = Path(args.out)
    (out / "providers.json").write_text(json.dumps(providers, indent=2) + "\n")
    (out / "patients.json").write_text(json.dumps(patients, indent=2) + "\n")
    print(f"Wrote {len(providers['providers'])} providers and {len(patients['patients'])} patients "
          f"to {out}/ (seed={args.seed})")


if __name__ == "__main__":
    main()

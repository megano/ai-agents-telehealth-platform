"""
Keyless tests for the care-matching eligibility filter + relaxation.

These cover the safety-critical deterministic core: the legal floor is never
relaxed, zero-result cases relax the *minimum* firm constraints, and hard
language constraints are enforced. No API key or langchain needed.

Run directly (no pytest required):   python agents/test_matching.py
Or under pytest:                      pytest agents/test_matching.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from care_matching_agent import RELAXABLE, eligibility_filter, target_roles  # noqa: E402
from clinic_data import get_patient, get_pathway, get_providers  # noqa: E402

PROVIDERS = get_providers()


def test_exact_match_has_no_relaxation():
    # Sofia (pat_002) has only soft prefs for ob_gyn, so an exact match exists.
    res = eligibility_filter(get_patient("pat_002"), "ob_gyn", PROVIDERS)
    assert res["eligible"], "expected eligible OB-GYNs licensed in TX"
    assert res["relaxed"] == [], "no hard constraints should mean no relaxation"
    assert not res["escalate"]


def test_language_hard_constraint_enforced():
    # Carmen (pat_020) requires Spanish. If matched exactly, every eligible OB
    # must speak Spanish; otherwise language must be the relaxed constraint.
    res = eligibility_filter(get_patient("pat_020"), "ob_gyn", PROVIDERS)
    assert not res["escalate"]
    if res["relaxed"]:
        assert any(t["category"] == "language" for t in res["relaxed"])
    else:
        assert all("Spanish" in p.get("languages", []) for p in res["eligible"]), \
            "exact match must honor the Spanish hard constraint"


def test_zero_results_triggers_minimal_relaxation():
    # Priya (pat_021): Hindi + evening pelvic-floor PT in NM. No NM pelvic PT
    # speaks Hindi, so the language constraint forces relaxation, but NM does
    # have pelvic PTs (floor non-empty), so this relaxes rather than escalates.
    res = eligibility_filter(get_patient("pat_021"), "pelvic_floor_pt", PROVIDERS)
    assert not res["escalate"], "NM has pelvic-floor PTs; floor is non-empty"
    assert res["eligible"], "relaxation should surface near-matches"
    cats = {t["category"] for t in res["relaxed"]}
    assert "language" in cats, "the unsatisfiable Hindi constraint must be relaxed"


def test_legal_floor_is_never_relaxed():
    # Whatever gets relaxed, it is only ever a firm constraint, never license or
    # accepting_new_patients.
    res = eligibility_filter(get_patient("pat_021"), "pelvic_floor_pt", PROVIDERS)
    for t in res["relaxed"]:
        assert t["category"] in RELAXABLE
        assert t["category"] not in ("license", "accepting_new_patients")


def test_no_floor_escalates_instead_of_relaxing():
    # WY is outside the clinic's license footprint, so no licensed pediatrician
    # exists there. That is an escalation (expand network), not a relaxation.
    synthetic = {"location": {"state": "WY"}, "preference_profile": []}
    res = eligibility_filter(synthetic, "pediatrician", PROVIDERS)
    assert res["floor_count"] == 0
    assert res["escalate"] and res["eligible"] == []


def test_relaxation_is_minimal_and_picks_the_constraint_that_works():
    # Two hard constraints: gender=female (cheap) and language=Hindi (costly).
    # No provider speaks Hindi, so dropping gender alone still yields nothing;
    # only dropping language works. The relaxer must drop *only* language.
    provs = [
        {"id": "t1", "specialty": "Therapy", "licensed_states": "all",
         "accepting_new_patients": True, "gender": "female", "languages": ["English"],
         "availability_dayparts": ["morning"], "hub": "hub_la", "virtual_available": True},
        {"id": "t2", "specialty": "Therapy", "licensed_states": "all",
         "accepting_new_patients": True, "gender": "male", "languages": ["English"],
         "availability_dayparts": ["morning"], "hub": None, "virtual_available": True},
    ]
    pat = {"location": {"state": "CA"}, "preference_profile": [
        {"category": "gender", "preference": "female", "applies_to": "therapist",
         "tier": "hard_constraint", "strength": "explicit", "weight": 0.7},
        {"category": "language", "preference": "Hindi", "applies_to": "therapist",
         "tier": "hard_constraint", "strength": "explicit", "weight": 1.0},
    ]}
    res = eligibility_filter(pat, "therapist", provs)
    assert {t["category"] for t in res["relaxed"]} == {"language"}, \
        "should relax only language, not the cheaper-but-useless gender drop"
    assert {p["id"] for p in res["eligible"]} == {"t1"}


def test_target_roles_match_the_stage():
    roles = target_roles(get_patient("pat_003"), get_pathway("pathway_standard_maternity"))
    assert "therapist" in roles, "postpartum stage must include therapist"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run() else 0)

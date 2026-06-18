"""
Preference-Weighted Care Matching Agent
=======================================
Ranks the clinic's eligible providers for a patient by that patient's *weighted*
preferences, producing a recommendation + rationale for a human coordinator.

Design (see ARCHITECTURE.html / PLAN.md):
  Eligibility filter (deterministic code, no LLM):
    - Hard legal floor, NEVER relaxed: state license, accepting_new_patients.
      Zero results here is an escalation (expand network / staff), not relaxation.
    - Relaxable firm constraints: language, scheduling daypart, gender, modality.
    - Progressive *minimal* relaxation on zero results: find the smallest, least
      costly set of relaxable constraints to drop so the eligible set is non-empty;
      every relaxation is tagged. The legal floor is never touched.
  Matcher (Sonnet, the one runnable LLM step):
    - Per-role weighted ranking of the eligible providers + a rationale each.
  Refinement (signals -> profile) is documented in the architecture but not run
  here; v1's only LLM call is the matcher.

The eligibility filter and relaxation are pure functions with no model and no
network, so they are unit-tested without an API key (see test_matching.py). The
LLM client is imported lazily inside the matcher so this module imports cleanly
without langchain installed.

Usage:
    python agents/care_matching_agent.py --patient pat_003            # rank full stage team
    python agents/care_matching_agent.py --patient pat_003 --role therapist
    python agents/care_matching_agent.py --patient pat_021 --no-llm   # keyless: filter + relaxation only
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Optional

from clinic_data import get_patient, get_pathway, get_providers, infer_provider_type

# ── Config ────────────────────────────────────────────────────────────────────

MATCHER_MODEL = os.getenv("MATCHER_MODEL", "claude-sonnet-4-6")
OUTPUT_DIR = Path(__file__).parent / "output"
STANDARD_PATHWAY = "pathway_standard_maternity"

# Relaxable firm constraints, in ascending "cost" of relaxing them. Lower cost is
# preferred to drop first. The legal floor (license, accepting_new_patients) is
# deliberately NOT in this list: it is never relaxed.
RELAX_COST = {"gender": 1, "modality": 2, "scheduling": 3, "language": 4}
RELAXABLE = set(RELAX_COST)


# ── Preference extraction ─────────────────────────────────────────────────────

def _applies(entry: dict, role: str) -> bool:
    """Does this preference apply to the given care-team role?"""
    a = entry.get("applies_to", "all")
    if a == "all":
        return True
    if isinstance(a, list):
        return role in a or "all" in a
    return a == role


def role_constraints(patient: dict, role: str, tier: str) -> list[dict]:
    """Preference entries of a given tier (hard_constraint | soft) that apply to
    this role and name a relaxable category."""
    out = []
    for e in patient.get("preference_profile", []):
        if e.get("tier") == tier and e.get("category") in RELAXABLE and _applies(e, role):
            out.append(e)
    return out


def soft_signals(patient: dict, role: str) -> list[dict]:
    """All soft preferences applying to this role, used by the matcher to rank.
    Includes non-relaxable soft categories like therapy method (modality_method)."""
    return [e for e in patient.get("preference_profile", [])
            if e.get("tier") == "soft" and _applies(e, role)]


# ── Constraint checks (deterministic) ─────────────────────────────────────────

def _meets(prov: dict, category: str, preference: str) -> bool:
    """Whether a provider satisfies one relaxable constraint."""
    if category == "language":
        return preference in prov.get("languages", [])
    if category == "gender":
        return prov.get("gender") == preference
    if category == "scheduling":
        return preference in prov.get("availability_dayparts", [])
    if category == "modality":
        if preference == "virtual":
            return bool(prov.get("virtual_available"))
        if preference == "in-person":
            return prov.get("hub") is not None
        return True
    return True


def _floor_ok(prov: dict, patient: dict, role: str) -> bool:
    """Hard legal floor: correct role, licensed in the patient's state, open panel.
    Never relaxed."""
    if infer_provider_type(prov) != role:
        return False
    lic = prov.get("licensed_states")
    if lic != "all" and patient["location"]["state"] not in (lic or []):
        return False
    return bool(prov.get("accepting_new_patients", False))


# ── Eligibility filter + progressive relaxation ───────────────────────────────

def eligibility_filter(patient: dict, role: str, providers: list[dict]) -> dict:
    """Return the eligible providers for one role, relaxing the minimum set of
    firm constraints if an exact match is impossible.

    Result keys:
      role, floor_count, eligible (list of provider dicts), relaxed (list of
      tags), escalate (bool), note.
    """
    floor = [p for p in providers if _floor_ok(p, patient, role)]
    if not floor:
        return {
            "role": role, "floor_count": 0, "eligible": [], "relaxed": [],
            "escalate": True,
            "note": f"No provider meets the legal floor for {role} "
                    f"(licensed in {patient['location']['state']} and accepting). "
                    f"Expand network or escalate to staff; the floor is never relaxed.",
        }

    hard = role_constraints(patient, role, "hard_constraint")
    active = {e["category"]: e["preference"] for e in hard}

    def passes(p, cats):
        return all(_meets(p, c, active[c]) for c in cats)

    exact = [p for p in floor if passes(p, active.keys())]
    if exact:
        return {"role": role, "floor_count": len(floor), "eligible": exact,
                "relaxed": [], "escalate": False, "note": ""}

    # Minimal relaxation: smallest subset of firm constraints to drop (ties broken
    # by lowest total cost) so the eligible set becomes non-empty. Dropping every
    # firm constraint falls back to the floor, which is non-empty here, so this
    # always resolves.
    cats = list(active)
    for size in range(1, len(cats) + 1):
        subsets = sorted(combinations(cats, size),
                         key=lambda s: sum(RELAX_COST[c] for c in s))
        for drop in subsets:
            keep = [c for c in cats if c not in drop]
            cand = [p for p in floor if passes(p, keep)]
            if cand:
                return {
                    "role": role, "floor_count": len(floor), "eligible": cand,
                    "relaxed": [_relax_tag(c, active[c]) for c in drop],
                    "escalate": False,
                    "note": "Exact match impossible; relaxed the minimum firm "
                            "constraints below. The legal floor was not touched.",
                }
    # Unreachable (dropping all firm constraints == floor, already non-empty).
    return {"role": role, "floor_count": len(floor), "eligible": floor,
            "relaxed": [_relax_tag(c, active[c]) for c in cats],
            "escalate": False, "note": "Relaxed all firm constraints."}


def _relax_tag(category: str, preference: str) -> dict:
    notes = {
        "language": f"requested {preference}; nearest matches may need an interpreter",
        "scheduling": f"requested {preference}; other dayparts offered",
        "gender": f"requested {preference} provider",
        "modality": f"requested {preference}",
    }
    return {"category": category, "requested": preference, "note": notes.get(category, "")}


def target_roles(patient: dict, pathway: dict) -> list[str]:
    """Roles the patient's current care stage matches against, from the pathway."""
    stage = patient.get("care_stage")
    sd = next((s for s in pathway["stages"] if s["id"] == stage), None)
    if not sd:
        return []
    seen, roles = set(), []
    for i in sd["interventions"]:
        t = i["provider_type"]
        if t not in seen:
            seen.add(t)
            roles.append(t)
    return roles


# ── LLM matcher (the one runnable model step) ─────────────────────────────────

def _provider_card(p: dict) -> dict:
    """Minimal provider view sent to the matcher: no DOB, no raw contact."""
    card = {
        "provider_id": p["id"], "name": p["name"], "gender": p.get("gender"),
        "languages": p.get("languages", []),
        "availability_dayparts": p.get("availability_dayparts", []),
        "virtual_available": p.get("virtual_available"),
        "in_person": p.get("hub") is not None,
        "avg_rating": p.get("avg_rating"),
    }
    if "modalities" in p:
        card["modalities"] = p["modalities"]
    return card


def rank_with_llm(patient: dict, role: str, eligible: list[dict],
                  relaxed: list[dict]) -> list[dict]:
    """Rank eligible providers by the patient's weighted soft preferences.
    Lazily imports langchain so the deterministic core needs neither it nor a key.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    class RankedMatch(BaseModel):
        provider_id: str = Field(description="Provider ID from the eligible list")
        rank: int = Field(description="1 = best match")
        score: float = Field(description="Match quality 0.0-1.0")
        rationale: str = Field(description="One sentence; cite which weighted preferences drove the score")

    class Ranking(BaseModel):
        matches: list[RankedMatch]

    signals = soft_signals(patient, role)
    system = (
        "You are the care-matching engine for Astralace Women's Health. Rank the "
        "ELIGIBLE providers for ONE care-team role by how well each fits THIS "
        "patient's weighted preferences. Higher weight means more important. "
        "Explicit (stated) preferences outrank inferred (behavioral) ones; never "
        "let an inferred signal override an explicit one. Preferences not "
        "applicable to this role carry no weight. If constraints were relaxed, the "
        "near-matches are imperfect by definition; rank by least compromise. Return "
        "every eligible provider exactly once."
    )
    user = json.dumps({
        "role": role,
        "patient_weighted_preferences": [
            {"category": s["category"], "preference": s["preference"],
             "weight": s["weight"], "strength": s["strength"]} for s in signals
        ],
        "relaxed_constraints": relaxed,
        "eligible_providers": [_provider_card(p) for p in eligible],
    }, indent=2)

    llm = ChatAnthropic(model=MATCHER_MODEL, max_tokens=2048, temperature=0.2)
    result: Ranking = llm.with_structured_output(Ranking).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)])
    ranked = sorted([m.model_dump() for m in result.matches], key=lambda m: m["rank"])
    return ranked


# ── Orchestration ─────────────────────────────────────────────────────────────

def match(patient_id: str, role: Optional[str] = None, use_llm: bool = True) -> dict:
    patient = get_patient(patient_id)
    providers = get_providers()
    pathway = get_pathway(STANDARD_PATHWAY)

    roles = [role] if role else target_roles(patient, pathway)
    if not roles:
        raise ValueError(f"No target roles for {patient_id} (care_stage="
                         f"{patient.get('care_stage')}). Pass --role explicitly.")

    results = []
    for r in roles:
        elig = eligibility_filter(patient, r, providers)
        block = {
            "role": r,
            "eligible_count": len(elig["eligible"]),
            "escalate": elig["escalate"],
            "relaxed": elig["relaxed"],
            "note": elig["note"],
        }
        if use_llm and elig["eligible"] and not elig["escalate"]:
            block["ranked"] = rank_with_llm(patient, r, elig["eligible"], elig["relaxed"])
        else:
            # Keyless / escalation path: list eligible providers unranked.
            block["eligible_providers"] = [_provider_card(p) for p in elig["eligible"]]
        results.append(block)

    return {"patient_id": patient_id, "patient_name": patient["name"],
            "care_stage": patient.get("care_stage"), "roles": results}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_summary(out: dict) -> None:
    print(f"\n{'='*64}\n🧭  CARE MATCHING, Astralace Women's Health\n{'='*64}")
    print(f"Patient: {out['patient_name']} ({out['patient_id']}) · stage {out['care_stage']}")
    for block in out["roles"]:
        print(f"\n── {block['role']} ──  eligible: {block['eligible_count']}")
        if block["escalate"]:
            print(f"  ⛔ ESCALATE: {block['note']}")
            continue
        if block["relaxed"]:
            for t in block["relaxed"]:
                print(f"  ⚠ relaxed {t['category']}: {t['note']}")
        if "ranked" in block:
            for m in block["ranked"][:5]:
                print(f"  {m['rank']}. {m['provider_id']}  score={m['score']:.2f}  {m['rationale']}")
        else:
            for c in block.get("eligible_providers", [])[:8]:
                print(f"  • {c['provider_id']}  {c['name']}  ({', '.join(c['languages'])}; "
                      f"{', '.join(c['availability_dayparts'])})")


def require_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set (needed for the LLM matcher).")
        print("  cp .env.example .env  →  set ANTHROPIC_API_KEY=sk-ant-...")
        print("Or run with --no-llm for the deterministic filter + relaxation only.")
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Astralace Care Matching Agent")
    ap.add_argument("--patient", default="pat_003", help="Patient ID (default pat_003)")
    ap.add_argument("--role", default=None, help="Match a single role (e.g. therapist); default = full stage team")
    ap.add_argument("--no-llm", action="store_true", help="Deterministic filter + relaxation only; no API key needed")
    args = ap.parse_args()

    use_llm = not args.no_llm
    if use_llm:
        require_api_key()
        # Load .env lazily; only relevant on the LLM path.
        try:
            from dotenv import load_dotenv
            load_dotenv()
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            require_api_key()
        except ImportError:
            pass

    try:
        out = match(patient_id=args.patient, role=args.role, use_llm=use_llm)
    except ValueError as e:
        if os.getenv("DEBUG"):
            raise
        print(f"\nERROR: {e}")
        raise SystemExit(1)

    _print_summary(out)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"match_{args.patient}{'_' + args.role if args.role else ''}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n💾 {path}")


if __name__ == "__main__":
    main()

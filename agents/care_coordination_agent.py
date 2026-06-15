"""
Care Coordination Agent
=======================
Event-driven, proactive care team planning for Astralace Women's Health.

Architecture:
  Transition Coordinator (TC): persistent LangGraph orchestrator, wakes on events
  Provider Filter: DB lookup, not an LLM (filters by state license, availability)
  Care Team Planner: Sonnet, reasons over candidates → CarePlanRecommendation
  Validator: Opus, hard checks → ValidationResult
  Care Team Patient Intro Drafter: Sonnet, writes patient-facing care team intro

Demo scenario:
  Sofia Nguyen (pat_002) · TX · 26 weeks (T2) → TC detects upcoming T2→T3 transition
  → Plans T3 provider team → Validates → Drafts patient intro

Usage:
    python care_coordination_agent.py --patient pat_002

Output:
    agents/output/care_plan_<patient_id>.json: agent decision log
    agents/output/care_intro_<patient_id>.txt: patient-facing care team intro
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# Load .env from the repo root, regardless of where the script is run from.
# Also picks up a .env in the current directory or any parent.
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
# Models are overridable via environment variables. Defaults: Sonnet for the
# writer/planner/drafter, Opus 4.8 for the validator (4.7 is legacy).

PLANNER_MODEL   = os.getenv("PLANNER_MODEL",   "claude-sonnet-4-6")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "claude-opus-4-8")
DRAFTER_MODEL   = os.getenv("DRAFTER_MODEL",   "claude-sonnet-4-6")

MAX_VALIDATION_RETRIES = 2

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MOCK_DATA_DIR = Path(__file__).parent.parent / "mock-data"

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_json(filename: str) -> dict:
    with open(MOCK_DATA_DIR / filename) as f:
        return json.load(f)


def get_patient(patient_id: str) -> dict:
    data = load_json("patients.json")
    for p in data["patients"]:
        if p["id"] == patient_id:
            return p
    valid_ids = ", ".join(p["id"] for p in data["patients"])
    raise ValueError(f"Patient '{patient_id}' not found. Available patient IDs: {valid_ids}")


def get_providers() -> list[dict]:
    return load_json("providers.json")["providers"]


def get_pathway(pathway_id: str) -> dict:
    data = load_json("care-pathways.json")
    for p in data["pathways"]:
        if p["id"] == pathway_id:
            return p
    raise ValueError(f"Pathway {pathway_id} not found")


# ── Structured outputs ────────────────────────────────────────────────────────

class RecommendedProvider(BaseModel):
    provider_id: str = Field(description="Provider ID from providers.json (e.g. prov_002)")
    provider_name: str = Field(description="Full name and credentials")
    provider_type: str = Field(description="Role in the care plan (e.g. ob_gyn, pelvic_floor_pt)")
    intervention: str = Field(description="Specific intervention this provider delivers in this stage")
    rationale: str = Field(description="Why this provider was selected, for clinic audit log, not patient-facing")
    continuity_required: bool = Field(description="Whether this provider must be retained in subsequent stages")


class CarePlanRecommendation(BaseModel):
    """Structured output from the Care Team Planner."""
    patient_id: str
    target_stage: str = Field(description="The care stage being planned (e.g. T3, postpartum)")
    recommended_providers: list[RecommendedProvider]
    unmet_needs: list[str] = Field(
        description="Provider types required by pathway that could not be filled from available candidates",
        default_factory=list
    )
    planner_notes: str = Field(description="Any caveats or flags for the validator")


class ValidationCheck(BaseModel):
    check: str
    passed: bool
    detail: str


class ValidationResult(BaseModel):
    """Structured output from the Validator."""
    passed: bool
    checks: list[ValidationCheck]
    failures: list[str] = Field(description="List of failed check names")
    fix_instructions: str = Field(
        description="Concrete instructions for the Planner to revise if passed=False. Empty if passed=True."
    )


# ── TC State ──────────────────────────────────────────────────────────────────

class TCState(BaseModel):
    """Shared state flowing through the LangGraph nodes."""
    # Input
    patient: dict
    pathway: dict
    target_stage: str
    candidate_providers: list[dict] = Field(default_factory=list)

    # Working state
    recommendation: Optional[CarePlanRecommendation] = None
    validation: Optional[ValidationResult] = None
    validation_attempts: int = 0

    # Output
    patient_intro: str = ""
    final_plan: Optional[CarePlanRecommendation] = None
    blocked: bool = False
    completed: bool = False


# ── LLM clients ───────────────────────────────────────────────────────────────

planner_llm = ChatAnthropic(
    model=PLANNER_MODEL,
    max_tokens=4096,
    temperature=0.3,
).with_structured_output(CarePlanRecommendation)

validator_llm = ChatAnthropic(
    model=VALIDATOR_MODEL,
    max_tokens=2048,
    temperature=0.0,
).with_structured_output(ValidationResult)

drafter_llm = ChatAnthropic(
    model=DRAFTER_MODEL,
    max_tokens=2048,
    temperature=0.7,
)

# ── System prompts ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are the Care Team Planner for Astralace Women's Health.

Your job is to select the right providers from a list of candidates to staff a patient's
upcoming care stage. You are building a proactive care team, the goal is to introduce
the right providers BEFORE problems arise, not after.

Selection rules:
1. Match provider specialty/type to what the care pathway requires for the target stage.
2. All providers must be licensed in the patient's state.
3. Respect patient preferences (gender, language) where candidates allow.
4. Preserve continuity: if a provider has continuity_required=true from the current stage,
   they must be carried forward unless explicitly unavailable.
5. Only select from accepting_new_patients=true providers.
6. If a required provider type has no valid candidate, flag it in unmet_needs, do not
   hallucinate a provider.

Return a structured CarePlanRecommendation.
"""

VALIDATOR_SYSTEM = """You are the Care Plan Validator for Astralace Women's Health.
You are a safety critic, your job is to catch errors before a care plan reaches a patient.

Run these checks on the recommended care plan:
1. state_license_valid: every recommended provider is licensed in the patient's state
2. accepting_new_patients: every provider has accepting_new_patients=true
3. no_hallucinated_ids: every provider_id exists in the provided candidate list
4. continuity_preserved: any provider flagged continuity_required in the current stage
                               is included in the recommendation (or explicitly noted as unavailable)
5. unmet_needs_flagged: if pathway requires a provider type with no candidate,
                               unmet_needs is non-empty (not silently skipped)
6. language_preference_met: if patient requires a specific language, at least one provider
                               per required type speaks that language (or unmet_needs flags it)

Return a ValidationResult. If any check fails, set passed=False and provide
concrete fix_instructions the Planner can act on immediately.
"""

DRAFTER_SYSTEM = """You are writing a warm, friendly care team introduction for a patient
at Astralace Women's Health. This message will be sent to the patient to introduce
their upcoming care team for their next stage of care.

Tone: warm, encouraging, human. Like a message from a trusted care coordinator.
Do NOT use clinical jargon. Do NOT explain agent logic or provider selection criteria.
Do mention each provider by name, their role in plain language, and a brief sentence
on how they'll support the patient specifically.
Keep it concise, under 300 words. Use short paragraphs, no bullet points.
"""

# ── Helper: Provider Filter (DB logic, not LLM) ───────────────────────────────

def filter_providers(patient: dict, target_stage_data: dict, all_providers: list[dict]) -> list[dict]:
    """
    Pure DB-style filter. No LLM involved.
    Returns providers that are:
      - Licensed in the patient's state (or state_license_required=False)
      - Accepting new patients
      - Relevant to at least one intervention type in the target stage
    """
    patient_state = patient["location"]["state"]
    required_types = {i["provider_type"] for i in target_stage_data["interventions"]}

    candidates = []
    for prov in all_providers:
        # State license check
        licensed = prov.get("licensed_states")
        if licensed != "all" and patient_state not in licensed:
            continue

        # Accepting new patients
        if not prov.get("accepting_new_patients", False):
            continue

        # Must serve at least one required provider type in this stage
        # We map specialty keywords to provider_type labels used in pathway
        ptype = _infer_provider_type(prov)
        if ptype in required_types:
            candidates.append({**prov, "_inferred_type": ptype})

    return candidates


def _infer_provider_type(prov: dict) -> str:
    """Map provider specialty string to the provider_type keys used in care-pathways.json."""
    specialty = prov.get("specialty", "").lower()
    mapping = {
        "obstetrics":            "ob_gyn",
        "prenatal":              "ob_gyn",
        "lactation":             "lactation_consultant",
        "infant feeding":        "lactation_consultant",
        "pelvic floor":          "pelvic_floor_pt",
        "nutrition":             "registered_dietitian",
        "dietitian":             "registered_dietitian",
        "mental health":         "therapist",
        "therapy":               "therapist",
        "therapist":             "therapist",
        "lcsw":                  "therapist",
        "lpc":                   "therapist",
        "psychiatry":            "psychiatrist",
        "pediatrics":            "pediatrician",
        "midwife":               "midwife",
        "career":                "career_coach",
        "return-to-work":        "career_coach",
        "maternity leave":       "career_coach",
        "health coach":          "health_coach",
        "fertility":             "fertility_specialist",
        "reproductive endocrin": "fertility_specialist",
        "menopause":             "menopause_specialist",
    }
    for keyword, ptype in mapping.items():
        if keyword in specialty:
            return ptype
    return "general"


# ── Nodes ─────────────────────────────────────────────────────────────────────

def provider_filter_node(state: TCState) -> TCState:
    """DB-style filter, no LLM. Narrows full provider list to valid candidates."""
    print(f"\n{'='*60}")
    print(f"🔎  PROVIDER FILTER  (state: {state.patient['location']['state']} · stage: {state.target_stage})")
    print(f"{'='*60}")

    all_providers = get_providers()
    target_stage_data = next(
        s for s in state.pathway["stages"] if s["id"] == state.target_stage
    )

    candidates = filter_providers(state.patient, target_stage_data, all_providers)

    print(f"Candidates found: {len(candidates)}")
    for c in candidates:
        print(f"  • {c['name']} ({c['_inferred_type']}), {c['location']['state'] if 'location' in c else c.get('licensed_states')}")

    return TCState(**{**state.model_dump(), "candidate_providers": candidates})


def care_team_planner_node(state: TCState) -> TCState:
    """Care Team Planner (Sonnet), selects and justifies provider team."""
    print(f"\n{'='*60}")
    attempt = state.validation_attempts + 1
    print(f"🧠  CARE TEAM PLANNER  (attempt {attempt}/{MAX_VALIDATION_RETRIES + 1})")
    print(f"{'='*60}")

    target_stage_data = next(
        s for s in state.pathway["stages"] if s["id"] == state.target_stage
    )

    prior_feedback = ""
    if state.validation and not state.validation.passed:
        prior_feedback = f"""
Your previous recommendation failed validation. Fix these issues:
{state.validation.fix_instructions}
"""

    user_prompt = f"""Plan the care team for the following patient's upcoming stage.

PATIENT:
- ID: {state.patient['id']}
- Name: {state.patient['name']}
- State: {state.patient['location']['state']}
- Language: {state.patient['language']}
- Current stage: {state.patient.get('care_stage', 'unknown')}
- Pregnancy week: {state.patient.get('pregnancy_week', 'N/A')}
- Preferences: {json.dumps(state.patient['matching_preferences'])}

TARGET STAGE: {state.target_stage}
Required interventions:
{json.dumps(target_stage_data['interventions'], indent=2)}

AVAILABLE CANDIDATES (already filtered for state license + availability):
{json.dumps([{{
    'id': p['id'],
    'name': p['name'],
    'specialty': p['specialty'],
    'type': p['_inferred_type'],
    'languages': p['languages'],
    'accepting_new_patients': p['accepting_new_patients']
}} for p in state.candidate_providers], indent=2)}
{prior_feedback}
Select the best provider for each required intervention. Return a CarePlanRecommendation.
"""

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    recommendation: CarePlanRecommendation = planner_llm.invoke(messages)

    print(f"Recommended {len(recommendation.recommended_providers)} providers:")
    for rp in recommendation.recommended_providers:
        print(f"  • {rp.provider_name} → {rp.intervention}")
    if recommendation.unmet_needs:
        print(f"  ⚠ Unmet needs: {recommendation.unmet_needs}")

    return TCState(**{**state.model_dump(), "recommendation": recommendation})


def validator_node(state: TCState) -> TCState:
    """Validator (Opus), hard checks on the care plan recommendation."""
    print(f"\n{'='*60}")
    print(f"✅  VALIDATOR")
    print(f"{'='*60}")

    # Build candidate ID set for hallucination check
    candidate_ids = {p["id"] for p in state.candidate_providers}

    # Identify continuity providers from current stage
    current_stage_id = state.patient.get("care_stage", "")
    continuity_providers = []
    if current_stage_id:
        current_stage_data = next(
            (s for s in state.pathway["stages"] if s["id"] == current_stage_id), None
        )
        if current_stage_data:
            continuity_types = [
                i["provider_type"] for i in current_stage_data["interventions"]
                if i.get("continuity_required")
            ]
            continuity_providers = [
                p for p in state.candidate_providers
                if p["_inferred_type"] in continuity_types
            ]

    user_prompt = f"""Validate this care plan recommendation.

PATIENT:
- State: {state.patient['location']['state']}
- Language: {state.patient['language']}
- Current stage: {state.patient.get('care_stage', 'unknown')}

RECOMMENDATION:
{json.dumps(state.recommendation.model_dump(), indent=2)}

VALID CANDIDATE IDs (for hallucination check):
{json.dumps(list(candidate_ids))}

CONTINUITY PROVIDERS (must be preserved if available):
{json.dumps([{{'id': p['id'], 'name': p['name'], 'type': p['_inferred_type']}} for p in continuity_providers])}

Run all 6 validation checks and return a ValidationResult.
"""

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    validation: ValidationResult = validator_llm.invoke(messages)

    print(f"Validation: {'PASS ✓' if validation.passed else 'FAIL ✗'}")
    for check in validation.checks:
        status = "✓" if check.passed else "✗"
        print(f"  {status} {check.check}: {check.detail}")

    return TCState(**{
        **state.model_dump(),
        "validation": validation,
        "validation_attempts": state.validation_attempts + 1,
    })


def care_team_intro_drafter_node(state: TCState) -> TCState:
    """Care Team Patient Intro Drafter (Sonnet), writes patient-facing intro."""
    print(f"\n{'='*60}")
    print(f"✍️  CARE TEAM PATIENT INTRO DRAFTER")
    print(f"{'='*60}")

    stage_names = {
        "T1": "First Trimester", "T2": "Second Trimester",
        "T3": "Third Trimester", "postpartum": "Postpartum"
    }
    stage_label = stage_names.get(state.target_stage, state.target_stage)

    providers_summary = "\n".join([
        f"- {rp.provider_name} ({rp.provider_type}): {rp.intervention}"
        for rp in state.recommendation.recommended_providers
    ])

    user_prompt = f"""Write a warm care team introduction for this patient.

PATIENT: {state.patient['name']}
UPCOMING STAGE: {stage_label}
CURRENT PREGNANCY WEEK: {state.patient.get('pregnancy_week', 'N/A')}

CARE TEAM FOR THIS STAGE:
{providers_summary}

Write a brief, warm message from their Astralace care coordinator introducing their
upcoming care team and what to expect. Address the patient by first name.
"""

    messages = [
        SystemMessage(content=DRAFTER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = drafter_llm.invoke(messages)
    intro = response.content

    print(f"Patient intro drafted ({len(intro)} chars)")
    print(f"\n{intro[:200]}...\n")

    return TCState(**{
        **state.model_dump(),
        "patient_intro": intro,
        "final_plan": state.recommendation,
        "completed": True,
    })


def block_node(state: TCState) -> TCState:
    """Validation failed after max retries, route to staff queue."""
    print(f"\n{'='*60}")
    print(f"🚫  BLOCKED, routing to staff queue")
    print(f"{'='*60}")
    print(f"Last failure: {state.validation.failures if state.validation else 'unknown'}")
    return TCState(**{**state.model_dump(), "blocked": True})


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_validation(state: TCState) -> str:
    if state.validation and state.validation.passed:
        return "draft"
    if state.validation_attempts >= MAX_VALIDATION_RETRIES:
        return "block"
    return "replan"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(TCState)

    graph.add_node("provider_filter",     provider_filter_node)
    graph.add_node("care_team_planner",   care_team_planner_node)
    graph.add_node("validator",           validator_node)
    graph.add_node("intro_drafter",       care_team_intro_drafter_node)
    graph.add_node("block",               block_node)

    graph.add_edge(START,                 "provider_filter")
    graph.add_edge("provider_filter",     "care_team_planner")
    graph.add_edge("care_team_planner",   "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "draft":   "intro_drafter",
            "replan":  "care_team_planner",
            "block":   "block",
        }
    )
    graph.add_edge("intro_drafter",       END)
    graph.add_edge("block",               END)

    return graph.compile()


# ── Output writers ────────────────────────────────────────────────────────────

def write_outputs(state: TCState, patient_id: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if state.blocked:
        blocked_path = OUTPUT_DIR / f"blocked_care_plan_{patient_id}_{timestamp}.json"
        with open(blocked_path, "w") as f:
            json.dump({
                "status": "blocked",
                "patient_id": patient_id,
                "target_stage": state.target_stage,
                "validation_attempts": state.validation_attempts,
                "last_validation": state.validation.model_dump() if state.validation else None,
                "last_recommendation": state.recommendation.model_dump() if state.recommendation else None,
                "blocked_at": timestamp,
                "action_required": "Staff review needed, care plan could not be validated automatically",
            }, f, indent=2)
        print(f"\n🚫 Blocked plan written → {blocked_path}")
        return

    # Care plan JSON (clinic-facing audit log)
    plan_path = OUTPUT_DIR / f"care_plan_{patient_id}_{timestamp}.json"
    with open(plan_path, "w") as f:
        json.dump({
            "status": "accepted",
            "patient_id": patient_id,
            "patient_name": state.patient["name"],
            "target_stage": state.target_stage,
            "validation_attempts": state.validation_attempts,
            "validation": state.validation.model_dump() if state.validation else None,
            "care_plan": state.final_plan.model_dump() if state.final_plan else None,
            "generated_at": timestamp,
        }, f, indent=2)

    # Patient intro (patient-facing)
    intro_path = OUTPUT_DIR / f"care_intro_{patient_id}_{timestamp}.txt"
    with open(intro_path, "w") as f:
        f.write(state.patient_intro)

    print(f"\n✅ Care plan  → {plan_path}")
    print(f"✅ Patient intro → {intro_path}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def require_api_key() -> None:
    """Fail fast with a friendly message if the API key is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("  1. cp .env.example .env")
        print("  2. edit .env  →  ANTHROPIC_API_KEY=sk-ant-...")
        print("Or export ANTHROPIC_API_KEY in your shell. Get a key at https://console.anthropic.com")
        raise SystemExit(1)


def run(patient_id: str, target_stage: Optional[str] = None) -> None:
    patient = get_patient(patient_id)

    # TC determines target stage from patient state if not overridden
    if not target_stage:
        stage_progression = {"T1": "T2", "T2": "T3", "T3": "postpartum"}
        current = patient.get("care_stage", "T1")
        target_stage = stage_progression.get(current, "postpartum")

    pathway = get_pathway("pathway_standard_maternity")

    print(f"\n{'='*60}")
    print(f"🏥  TRANSITION COORDINATOR, Astralace Women's Health")
    print(f"{'='*60}")
    print(f"Patient:       {patient['name']} ({patient_id})")
    print(f"Current stage: {patient.get('care_stage', 'unknown')} "
          f"(week {patient.get('pregnancy_week', 'N/A')})")
    print(f"Target stage:  {target_stage}")
    print(f"Pathway:       {pathway['name']}")

    initial_state = TCState(
        patient=patient,
        pathway=pathway,
        target_stage=target_stage,
    )

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    final_state = TCState(**final_state)

    write_outputs(final_state, patient_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astralace Care Coordination Agent")
    parser.add_argument("--patient", default="pat_002", help="Patient ID (default: pat_002)")
    parser.add_argument("--stage",   default=None,      help="Override target stage (T1/T2/T3/postpartum)")
    args = parser.parse_args()

    require_api_key()

    # Known user errors (e.g. bad patient id) print a clean message. Set DEBUG=1
    # for the full traceback. Unexpected errors are NOT caught here: they raise
    # with a full traceback by design, since that's what you need to debug them.
    try:
        run(patient_id=args.patient, target_stage=args.stage)
    except ValueError as e:
        if os.getenv("DEBUG"):
            raise
        print(f"\nERROR: {e}")
        print("(set DEBUG=1 for the full traceback)")
        raise SystemExit(1)

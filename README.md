# AI Agents for Telehealth: Proactive Women's Health Care

> **This is not a real company.** Astralace Women's Health is a fictional telehealth platform.
> All patient data, provider data, and clinical scenarios are entirely synthetic.
> Nothing in this repo constitutes medical advice.

---

## What if AI agents made women's health care proactive instead of reactive?

This demo focuses on the pregnancy and postpartum journey as the first user journey: a high-stakes, time-sensitive arc where the difference between proactive and reactive care is most concrete.

Most telehealth platforms wait for patients to show up with a problem. This project explores a different model: what if an AI system anticipated the next stage of a patient's journey and built her care team *before* she needed it?

The lactation consultant is introduced at week 34, before birth, not after a struggling new mother calls in a panic. The pelvic floor PT starts at week 20, before symptoms appear. The career coach enters at week 10, while there's still time to design a thoughtful maternity leave plan with her manager.

This repo is a portfolio demo that uses a fictional Maven-style women's telehealth clinic to showcase:

- **Proactive care coordination**: event-driven agents that plan ahead, not triage after
- **Multi-agent system design**: LangGraph orchestration, structured outputs, validator-as-safety-critic
- **Product thinking**: care pathway design, clinical workflow constraints, data modeling

---

## The Care Coordination Agent [`agents/care_coordination_agent.py`]

Event-driven care team planning for maternity and postpartum journeys. The Transition Coordinator (TC) wakes on stage milestone events and orchestrates a four-node pipeline.

```mermaid
flowchart TD
  EV["⚡ Event: stage_milestone\nSofia · TX · week 26"] --> TC["🏥 Transition Coordinator\nLangGraph persistent state"]
  TC --> PF["🔎 Provider Filter\nDB lookup, not an LLM"]
  PF -->|candidates| CP["🧠 Care Team Planner\nSonnet · CarePlanRecommendation"]
  CP --> VA["✅ Validator\nOpus · 6 hard checks · ValidationResult"]
  VA -->|pass| DR["✍️ Care Team Patient Intro Drafter\nSonnet · patient-facing"]
  DR --> OUT["💾 care_plan_*.json\ncare_intro_*.txt"]
  VA -->|"fail · retry ≤ 2x"| CP
  VA -->|max retries exceeded| BL["⛔ Staff Queue\nblocked_*.json"]

  style PF stroke-dasharray:5
  style BL fill:#fdecea,stroke:#c0392b
```

**Demo scenario:** Sofia Nguyen, TX, 26 weeks pregnant, second trimester. The TC detects she's approaching her third trimester and builds her T3 care team.

```bash
python agents/care_coordination_agent.py --patient pat_002
```

**Design decisions worth noting:**

*Provider Filter is not an LLM.* State licensure checks and availability filters are deterministic. Running them as a DB-style function before the Planner reduces token cost, removes hallucination surface area, and keeps the model focused on reasoning, not lookups.

*Validator blocks, not retries forever.* After 2 failed attempts, the plan routes to a staff queue rather than degrading silently. In a system touching patient care decisions, "probably correct" isn't acceptable.

**Validator hard checks:**

| Check | What it catches |
|-------|----------------|
| `state_license_valid` | Provider not licensed in patient's state |
| `accepting_new_patients` | Closed panel slipped through filter |
| `no_hallucinated_ids` | Planner invented a provider not in candidate list |
| `continuity_preserved` | Required continuity provider dropped without explanation |
| `unmet_needs_flagged` | Required provider type silently missing from plan |
| `language_preference_met` | Patient language need unmet |

---

## Care Pathway Design

The standard maternity pathway spans four stages. Each intervention has an `evidence_level` (`established` or `emerging`): the pathway combines clinical standards with proactive interventions designed to prevent problems rather than respond to them.

| Stage | Weeks | Proactive care team |
|-------|-------|---------------------|
| T1 | 1-12 | OB-GYN · Dietitian · Therapist · Career Coach *(leave planning)* |
| T2 | 13-27 | OB-GYN · Pelvic Floor PT · Dietitian *(GD screening)* · Career Coach *(handoff)* |
| T3 | 28-40 | OB-GYN · Pelvic Floor PT · Lactation Consultant · Pediatrician · Career Coach *(RTW plan)* |
| Postpartum | 0-12pp | OB-GYN · LC · Pelvic Floor PT · Therapist · Career Coach *(re-entry)* |

The data layer also includes schema stubs for future pathway variants (Gestational Diabetes, Mental Health Risk, High-Risk OB), each triggered by `clinician_flagged_need`, never auto-activated by the agent.

---

## Mock Data

All data is synthetic. Located in `mock-data/`:

| File | Contents |
|------|----------|
| `providers.json` | 14 providers across specialties: OB-GYN, therapist, lactation consultant (LC), pelvic floor PT, dietitian, career coach, psychiatrist, and more. Each has `licensed_states`, languages, and insurance. |
| `patients.json` | 7 patients across care areas. Maternity patients include `care_stage` and `pregnancy_week`. |
| `care-pathways.json` | Standard maternity pathway (4 stages, 20+ interventions) + future variant schemas. |
| `schedules.json` | Provider availability and booked slots. |
| `clinic.json` | Clinic metadata: hubs, product areas, policies. |

---

## Architecture

[`ARCHITECTURE.html`](ARCHITECTURE.html): full flow diagrams, data access boundaries table, and validator pattern. Open in a browser; diagrams are rendered with Mermaid.js.

---

## Tech Stack

| | |
|--|--|
| **Orchestration** | LangGraph: StateGraph, conditional routing, persistent TC state |
| **Models** | `claude-sonnet-4-6` (writer / planner / drafter) · `claude-opus-4-8` (reviewer / validator) |
| **Structured outputs** | Pydantic: `ReviewResult`, `CarePlanRecommendation`, `ValidationResult` |
| **LLM client** | langchain-anthropic: `.with_structured_output()` for typed model responses |
| **Runtime** | Python 3.11+ |

---

## Setup

```bash
# Install dependencies
pip install -r agents/requirements.txt

# Add your Anthropic API key
cp .env.example .env
# edit .env → ANTHROPIC_API_KEY=sk-...

# Run the care coordination agent (demo: Sofia, TX, 26 weeks)
python agents/care_coordination_agent.py --patient pat_002
```

**Model selection (optional).** The models are configurable via environment
variables, so you can run the demo on whatever you have access to:

| Variable | Default | Role |
|----------|---------|------|
| `PLANNER_MODEL`   | `claude-sonnet-4-6` | Care Team Planner |
| `VALIDATOR_MODEL` | `claude-opus-4-8`   | Validator (safety critic) |
| `DRAFTER_MODEL`   | `claude-sonnet-4-6` | Patient Intro Drafter |

Set them in `.env` or your shell to override.

---

## Project Structure

```
├── agents/
│   ├── care_coordination_agent.py   # Care coordination + Transition Coordinator
│   └── output/                      # Care plans + patient intros per run
├── mock-data/
│   ├── providers.json               # 14 providers with licensed_states
│   ├── patients.json                # 7 patients with care_stage
│   ├── care-pathways.json           # Standard maternity pathway + variant schemas
│   ├── schedules.json
│   └── clinic.json
└── ARCHITECTURE.html                # Rendered architecture diagrams (open in browser)
```

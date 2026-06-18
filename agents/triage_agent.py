"""
Patient-Message Triage Agent
============================
One triage pass over the inbound message queue, two risk-stratified outputs:

  Shared classifier (Sonnet, one pass):   each message -> acuity tier + category + rationale
  Escalation guard (deterministic code):   red-flag patterns force top priority and can
                                            NEVER be downgraded by the model
  Handler A, Inbox Prioritizer:            reprioritized MD worklist (most acute first)
  Handler B, Refill Router (isolated):     medication messages -> refill_guard -> RN vs MD queue

Perception is shared (classify each message once); the two decision/action handlers
are separate so the higher-risk refill logic is bounded and independently testable.
Escalations and RN-routed items use the same staff-queue artifact shape the
care-coordination agent emits, so the two agents compose into one system.

The escalation guard and the refill guard are pure functions, tested without an
API key (test_triage.py). The only model call is the shared classifier, imported
lazily so the deterministic core needs neither langchain nor a key.

Usage:
    python agents/triage_agent.py                 # full LLM triage of the inbox
    python agents/triage_agent.py --no-llm         # keyless: guards only, no API key
"""

import argparse
import json
import os
import re
from pathlib import Path

from clinic_data import MOCK_DATA_DIR, get_patient, load_json
from refill_guard import is_medication_request, reason_text, refill_decision

CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "claude-sonnet-4-6")
OUTPUT_DIR = Path(__file__).parent / "output"

ACUITY_ORDER = {"urgent": 0, "same-day": 1, "routine": 2, "unclassified": 3}

# Perinatal red-flag patterns. If any matches the raw message text, the message
# is forced to top priority and tagged "clinical review required". The model
# cannot override this; the guard runs on raw text, after classification, and
# only ever escalates.
ESCALATION_PATTERNS = [
    (r"harm(ing)? (myself|the baby)", "self_harm"),
    (r"hurt(ing)? myself", "self_harm"),
    (r"\bsuicidal\b|kill myself|end my life", "self_harm"),
    (r"the baby move|felt the baby move|baby hasn'?t moved|decreased.*movement", "decreased_fetal_movement"),
    (r"seeing spots|flashing lights|blurred vision|vision changes", "preeclampsia_signs"),
    (r"soaking (through )?(a )?pad|heavy bleeding|bleeding heavily|hemorrhage", "hemorrhage"),
    (r"chest pain|chest tightness|trouble breathing|can'?t breathe|shortness of breath|difficulty breathing", "cardiopulmonary"),
    (r"severe abdominal pain|water broke|preterm", "obstetric_emergency"),
]


# ── Deterministic escalation guard ────────────────────────────────────────────

def escalation_guard(body: str) -> dict:
    """Scan raw message text for red flags. Returns {escalate, reason}. Only ever
    escalates; it never lowers a message's priority."""
    text = body.lower()
    for pattern, label in ESCALATION_PATTERNS:
        if re.search(pattern, text):
            return {"escalate": True, "reason": label}
    return {"escalate": False, "reason": None}


# ── Shared classifier (the one model step) ────────────────────────────────────

def classify_inbox_llm(messages: list[dict]) -> dict:
    """Classify every message in one batched call. Returns {id: {acuity, category,
    rationale}}. Lazy langchain import keeps the deterministic core key-free."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    class MsgClass(BaseModel):
        id: str
        acuity: str = Field(description="one of: urgent, same-day, routine")
        category: str = Field(description="one of: urgent_symptom, clinical_question, refill, scheduling, billing_admin")
        rationale: str = Field(description="one short clause")

    class InboxClass(BaseModel):
        classifications: list[MsgClass]

    system = (
        "You triage an inbound patient-message inbox for a perinatal telehealth "
        "clinic. For EACH message assign an acuity (urgent / same-day / routine) "
        "and a category (urgent_symptom, clinical_question, refill, scheduling, "
        "billing_admin). A deterministic safety guard runs after you and will "
        "independently escalate true red flags, so do not rely on yourself to "
        "catch every emergency, but do flag obvious ones as urgent. Classify every "
        "message exactly once."
    )
    payload = [{"id": m["id"], "body": m["body"]} for m in messages]
    llm = ChatAnthropic(model=CLASSIFIER_MODEL, max_tokens=2048, temperature=0.1)
    result: InboxClass = llm.with_structured_output(InboxClass).invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(payload, indent=2)),
    ])
    return {c.id: c.model_dump() for c in result.classifications}


# ── Orchestration ─────────────────────────────────────────────────────────────

def triage(use_llm: bool = True) -> dict:
    data = load_json("messages.json")
    messages = data["messages"]
    protocol = data["rn_refillable_protocol"]["medications"]

    classed = classify_inbox_llm(messages) if use_llm else {}

    md_worklist, rn_queue, admin_queue, staff_queue = [], [], [], []

    for m in messages:
        esc = escalation_guard(m["body"])
        cls = classed.get(m["id"], {})
        # The guard wins: an escalated message is urgent no matter what the model said.
        acuity = "urgent" if esc["escalate"] else cls.get("acuity", "unclassified")
        category = ("urgent_symptom" if esc["escalate"]
                    else cls.get("category")
                    or ("refill" if is_medication_request(m["body"]) else "unclassified"))

        item = {"message_id": m["id"], "patient_id": m["patient_id"],
                "acuity": acuity, "category": category,
                "subject": m.get("subject"), "rationale": cls.get("rationale", "")}

        if esc["escalate"]:
            item["flag"] = "clinical review required"
            item["escalation_reason"] = esc["reason"]
            staff_queue.append({
                "status": "escalated", "message_id": m["id"],
                "patient_id": m["patient_id"], "reason": esc["reason"],
                "action_required": "clinical review required",
            })
            md_worklist.append(item)
            continue

        # Refill Router (isolated handler) for medication messages.
        if category == "refill" or is_medication_request(m["body"]):
            decision = refill_decision(m, _safe_patient(m["patient_id"]), protocol)
            item["refill"] = {**decision, "reason_text": reason_text(decision["block_reason"])}
            if decision["rn_eligible"]:
                rn_queue.append(item)
            else:
                md_worklist.append(item)
            continue

        if category in ("scheduling", "billing_admin"):
            admin_queue.append(item)
        else:
            md_worklist.append(item)

    md_worklist.sort(key=lambda i: (ACUITY_ORDER.get(i["acuity"], 3), i["message_id"]))

    return {
        "inbox_size": len(messages),
        "md_worklist": md_worklist,
        "rn_queue": rn_queue,
        "admin_queue": admin_queue,
        "staff_queue": staff_queue,
    }


def _safe_patient(pid: str) -> dict:
    try:
        return get_patient(pid)
    except ValueError:
        return {"id": pid, "visit_history": []}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_summary(out: dict) -> None:
    print(f"\n{'='*64}\n📥  PATIENT-MESSAGE TRIAGE, Astralace Women's Health\n{'='*64}")
    print(f"Inbox: {out['inbox_size']} messages\n")
    print(f"── MD worklist (most acute first): {len(out['md_worklist'])} ──")
    for i in out["md_worklist"]:
        flag = f"  ⚑ {i['flag']}" if i.get("flag") else ""
        rf = ""
        if "refill" in i:
            rf = f"  [refill→MD: {i['refill']['block_reason']}]"
        print(f"  [{i['acuity']:>11}] {i['message_id']}  {i['category']}{rf}{flag}")
    print(f"\n── RN refill queue: {len(out['rn_queue'])} ──")
    for i in out["rn_queue"]:
        print(f"  {i['message_id']}  RN-eligible protocol refill"
              f"{'  (auto-approve flagged)' if i['refill'].get('auto_approve_eligible') else ''}")
    print(f"\n── Admin queue: {len(out['admin_queue'])} ──")
    for i in out["admin_queue"]:
        print(f"  {i['message_id']}  {i['category']}")
    print(f"\n── Escalations to staff queue: {len(out['staff_queue'])} ──")
    for s in out["staff_queue"]:
        print(f"  ⛔ {s['message_id']}  {s['reason']}  → {s['action_required']}")


def require_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set (needed for the classifier).")
        print("  cp .env.example .env  →  set ANTHROPIC_API_KEY=sk-ant-...")
        print("Or run with --no-llm for the deterministic guards only.")
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Astralace Patient-Message Triage Agent")
    ap.add_argument("--no-llm", action="store_true",
                    help="Deterministic guards only (escalation + refill); no API key needed")
    args = ap.parse_args()

    use_llm = not args.no_llm
    if use_llm:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        except ImportError:
            pass
        require_api_key()

    out = triage(use_llm=use_llm)
    _print_summary(out)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "triage_inbox.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n💾 {path}")


if __name__ == "__main__":
    main()

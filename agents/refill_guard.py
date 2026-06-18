"""
Refill safety guard, deterministic and isolated.

This is the higher-risk half of the triage agent, deliberately kept in its own
module with its own guard, audit, and autonomy config so it can be tested and
toggled independently of the inbox prioritizer. A bug here cannot degrade the
prioritizer's ranking.

The guard decides ONE thing for a medication/refill message: may it be routed to
the RN queue under standing orders, or must it stay with the prescriber? It never
fills or authorizes anything; routing is a worklist placement, the human decides.

Production note: in a real deployment, renewal authorization requires prescriber
sign-off, RN scope is set by state nurse-practice acts and clinic standing orders,
and the protocol list is owned by clinical leadership. This demo models that as a
configurable protocol list and a default-off autonomy ceiling, not a legal claim.
"""

import os
import re

# Autonomy ceiling, default L1. Never let the model raise this; it is clinic
# config only.
#   L0  human authorizes everything (no auto-routing)
#   L1  auto-route a clean protocol refill to the RN *queue*; human authorizes  [default]
#   L2  flag auto_approve_eligible on a clean protocol refill (still human-gated unless
#       a separate action toggle is on); off by default
AUTONOMY_LEVEL = os.getenv("REFILL_AUTONOMY", "L1")

_DOSE_CHANGE = [
    r"\bincrease\b", r"\bdecrease\b", r"higher dose", r"lower dose",
    r"adjust.*dose", r"change.*dose", r"up the dose", r"\bto \d+\s?mg\b",
]
_NEW_MEDICATION = [
    r"new prescription", r"start (me )?on", r"\bnew med", r"first time",
    r"never taken", r"begin taking", r"put me on",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def is_medication_request(body: str) -> bool:
    """Heuristic gate for the keyless demo: does this message look like a med /
    refill request? In the live pipeline the LLM classifier decides the category;
    this stand-in lets the deterministic path run without a model."""
    b = body.lower()
    return ("refill" in b or "prescription" in b or "start me on" in b
            or re.search(r"\d+\s?mg", b) is not None)


def has_qualifying_visit(patient: dict) -> bool:
    """A prescriber visit on file is the minimum context for an RN protocol
    refill. Modeled here as any documented visit."""
    return bool(patient.get("visit_history"))


def med_on_protocol(body: str, protocol_meds: list[str]) -> bool:
    b = body.lower()
    return any(med.lower() in b for med in protocol_meds)


def refill_decision(message: dict, patient: dict, protocol_meds: list[str]) -> dict:
    """Decide RN-eligible vs prescriber-required for a refill/medication message.

    Checks run in priority order; the first failing check is the block reason:
      dose_change -> new_medication -> off_protocol_med -> no_qualifying_visit.
    A message passing all checks is a straight protocol refill: RN-eligible.
    """
    body = message.get("body", "").lower()

    if _matches(_DOSE_CHANGE, body):
        reason = "dose_change"
    elif _matches(_NEW_MEDICATION, body):
        reason = "new_medication"
    elif not med_on_protocol(body, protocol_meds):
        reason = "off_protocol_med"
    elif not has_qualifying_visit(patient):
        reason = "no_qualifying_visit"
    else:
        reason = None

    rn_eligible = reason is None
    return {
        "message_id": message.get("id"),
        "rn_eligible": rn_eligible,
        "block_reason": reason,
        "route": "rn_queue" if rn_eligible else "md_queue",
        # Eligibility is not action. auto_approve only ever surfaces at L2, and even
        # then it is a flag for a human-gated step, never an auto-fill.
        "auto_approve_eligible": rn_eligible and AUTONOMY_LEVEL == "L2",
        "autonomy_level": AUTONOMY_LEVEL,
    }


_REASON_TEXT = {
    "dose_change": "Dose change requested; requires prescriber judgment.",
    "new_medication": "New medication / first fill; requires prescriber review.",
    "off_protocol_med": "Medication is not on the RN standing-order protocol list.",
    "no_qualifying_visit": "No qualifying recent visit on file; needs a prescriber touch first.",
    None: "Straight protocol refill; eligible for RN standing-order handling.",
}


def reason_text(block_reason) -> str:
    return _REASON_TEXT.get(block_reason, "")

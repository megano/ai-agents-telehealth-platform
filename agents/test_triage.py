"""
Keyless tests for the triage agent's deterministic guards.

Covers the safety-critical behavior: the escalation guard catches every seeded
red flag and no routine message, the refill guard routes exactly per the standing
-order rules, urgent messages float to the top and can never be downgraded, and
an escalated message never lands in the RN auto-route queue. No API key needed.

Run directly:   python agents/test_triage.py
Under pytest:   pytest agents/test_triage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clinic_data import get_patient, load_json  # noqa: E402
from refill_guard import refill_decision  # noqa: E402
from triage_agent import ACUITY_ORDER, escalation_guard, triage  # noqa: E402

DATA = load_json("messages.json")
MESSAGES = DATA["messages"]
PROTOCOL = DATA["rn_refillable_protocol"]["medications"]


def test_escalation_guard_matches_ground_truth():
    # Every seeded red flag escalates; every routine message does not. This is
    # the no-false-negative / no-false-positive check on the guard.
    for m in MESSAGES:
        got = escalation_guard(m["body"])["escalate"]
        exp = m["expected"]["escalation"]
        assert got == exp, f"{m['id']}: escalate={got}, expected {exp} ({m['expected']['note']})"


def test_refill_routing_matches_ground_truth():
    for m in MESSAGES:
        exp = m["expected"]["refill"]
        if exp is None:
            continue
        d = refill_decision(m, get_patient(m["patient_id"]), PROTOCOL)
        assert d["rn_eligible"] == exp["rn_eligible"], f"{m['id']}: rn_eligible mismatch"
        assert d["block_reason"] == exp["block_reason"], \
            f"{m['id']}: reason {d['block_reason']} != {exp['block_reason']}"


def test_buried_urgent_floats_and_cannot_be_downgraded():
    # msg_001 hides self-harm ideation inside a routine reschedule. The guard runs
    # on raw text, so it escalates regardless of the surface intent.
    out = triage(use_llm=False)
    urgent = {i["message_id"] for i in out["md_worklist"] if i["acuity"] == "urgent"}
    staff = {s["message_id"] for s in out["staff_queue"]}
    assert "msg_001" in urgent, "buried self-harm must be urgent"
    assert "msg_001" in staff, "escalation must reach the staff queue"


def test_md_worklist_is_acuity_sorted():
    out = triage(use_llm=False)
    order = [ACUITY_ORDER[i["acuity"]] for i in out["md_worklist"]]
    assert order == sorted(order), "MD worklist must be sorted most-acute-first"


def test_rn_queue_only_holds_clean_protocol_refills():
    out = triage(use_llm=False)
    for i in out["rn_queue"]:
        assert i["refill"]["rn_eligible"] is True
        assert i["refill"]["block_reason"] is None


def test_escalations_never_auto_route_to_rn():
    out = triage(use_llm=False)
    staff = {s["message_id"] for s in out["staff_queue"]}
    rn = {i["message_id"] for i in out["rn_queue"]}
    assert staff.isdisjoint(rn), "an escalated message must never be RN-auto-routed"


def test_dose_change_and_new_med_blocked_from_rn():
    out = triage(use_llm=False)
    blocked = {i["message_id"]: i["refill"]["block_reason"]
               for i in out["md_worklist"] if "refill" in i}
    assert blocked.get("msg_006") == "dose_change"
    assert blocked.get("msg_007") == "new_medication"


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

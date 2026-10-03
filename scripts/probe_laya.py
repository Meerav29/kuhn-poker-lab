# scripts/probe_laya.py
"""Prints (and saves) Laya's raw predict() output for one choice and one
noul question, so LayaScorer is written against reality, not the docs."""
import json
import os

import laya

agent = laya.load("convaiinnovations/laya")
state = "card=K history=b"
questions = {
    "act": {"type": "choice", "instructions": "The opponent bet. Do you fold or call?",
            "criteria": ["fold", "call"]},
    "call_yes": {"type": "noul", "instructions": "The opponent bet. Should you call?"},
}
result = agent.predict(state, questions)
text = json.dumps(result, indent=2, default=str)
print(text)

os.makedirs("tests/fixtures", exist_ok=True)
with open("tests/fixtures/laya_raw.json", "w", encoding="utf-8") as f:
    f.write(text)

"""Seeded, paired support decisions for the additional V4 study.

These are draft synthetic cases, not reviewed real-world benchmark labels.
Each base scenario and both of its variants belong to one partition.
"""
import random

import pandas as pd

from .contracts import canonical, digest

DATASET = "Support Policy"
FAMILIES = ("negation", "exception", "chronology", "paraphrase")
POLICY = (
    "Approve a refund if cancellation was recorded before dispatch. Otherwise, "
    "approve only when the cancellation request is within 30 days of purchase "
    "(day 30 is included) and either the item is defective or both the item is "
    "unopened and the order is not final sale. Deny all other refund requests. "
    "An order reference and the customer's contact preference do not affect eligibility."
)


def refund_label(facts):
    """Executable labeler; only the facts rendered in the observation are used."""
    return int(facts["cancel_before_dispatch"] or (
        facts["age_days"] <= 30 and (
            facts["defective"] or (facts["unopened"] and not facts["final_sale"]))))


def render(facts, reference, wording):
    yes_no = lambda value: "yes" if value else "no"
    cancel = "09:00" if facts["cancel_before_dispatch"] else "16:00"
    prefix = f"Order reference {reference}. Purchase was on day 0. "
    chronology = (f"On day {facts['age_days']}, dispatch was recorded at 14:00 "
                  f"and cancellation was recorded at {cancel}. ")
    if wording == 0:
        description = (f"The item is {'defective' if facts['defective'] else 'not defective'}. "
                       f"The package is {'unopened' if facts['unopened'] else 'not unopened'}. "
                       f"The order is {'final sale' if facts['final_sale'] else 'not final sale'}.")
    elif wording == 1:
        description = (f"Defect confirmed: {yes_no(facts['defective'])}. "
                       f"Unopened: {yes_no(facts['unopened'])}. "
                       f"Final-sale restriction: {yes_no(facts['final_sale'])}. "
                       "The customer prefers email replies.")
    elif wording == 2:
        description = (f"Inspection {'found' if facts['defective'] else 'did not find'} a defect. "
                       f"The parcel {'remains sealed' if facts['unopened'] else 'has been opened'}. "
                       f"The receipt {'marks' if facts['final_sale'] else 'does not mark'} this as final sale.")
    else:
        description = (f"This {'was' if facts['final_sale'] else 'was not'} a final-sale purchase. "
                       f"The goods {'have a defect' if facts['defective'] else 'have no defect'}; "
                       f"their packaging {'has never been opened' if facts['unopened'] else 'was opened'}. "
                       "Please use email for updates.")
    return prefix + chronology + description


def scenario_pair(family, held_out, rng, index):
    facts = {"age_days": rng.randint(10, 30), "cancel_before_dispatch": False,
             "defective": False, "final_sale": False, "unopened": False}
    if family == "negation":
        # Development: opening changes eligibility. Held-out combinations add
        # a defect exception or final-sale restriction that overrides opening.
        if held_out:
            facts.update(defective=index % 2 == 0, final_sale=index % 2 != 0)
        changed = "unopened"
    elif family == "exception":
        facts["final_sale"] = True
        if held_out:
            if index % 2:
                facts["age_days"] = rng.randint(31, 45)
            else:
                facts.update(final_sale=False, unopened=True)
        changed = "defective"
    elif family == "chronology":
        if held_out:
            facts["unopened"] = True
        changed = "cancel_before_dispatch"
    else:
        if held_out:
            facts.update(defective=True, final_sale=True,
                         age_days=rng.randint(31, 45) if index % 2 else rng.randint(10, 30))
        else:
            facts["unopened"] = index % 2 == 0
        changed = None
    other = dict(facts)
    if changed:
        other[changed] = not facts[changed]
    return facts, other


def generate(config):
    rng = random.Random(config["generator_seed"])
    rows = []
    for partition, count in config["pairs_per_partition"].items():
        divisor = 8 if partition == "test" else 4
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count % divisor:
            raise ValueError(f"{partition} pair count must be a positive multiple of {divisor}")
        for index in range(count):
            family = FAMILIES[index % len(FAMILIES)]
            family_index = index // len(FAMILIES)
            held_out = partition == "test" and family_index >= count // 8
            composition = "held-out" if held_out else "familiar"
            pair_id = digest({"seed": config["generator_seed"], "partition": partition, "pair": index})
            first, second = scenario_pair(family, held_out, rng, family_index)
            variants = [first, second]
            rng.shuffle(variants)
            relation = "preserve" if refund_label(first) == refund_label(second) else "flip"
            for member, facts in enumerate(variants):
                wording = (2 if held_out else 0) + (member if family == "paraphrase" else 0)
                rows.append({"text": render(facts, pair_id[:12], wording),
                             "label": refund_label(facts), "_pair_id": pair_id,
                             "_partition": partition, "_family": family,
                             "_composition": composition, "_pair_relation": relation,
                             "_facts": canonical(facts)})
    metadata = {"name": DATASET, "kind": "text", "features": ["text"],
                "labels": ["Deny the refund", "Approve the refund"], "task": POLICY,
                "source": "seeded executable support policy; generator version 1",
                "suite": "policy", "label_status": "synthetic-draft-needs-human-review"}
    return pd.DataFrame(rows), metadata




from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


route2 = load_script("route2_aop_summary", "05_run_aop_ppi_and_summary.py")
family_display = load_script("route2_family_display", "06_build_route2_family_display.py")


class EventJointHitTests(unittest.TestCase):
    def test_required_probability_boundaries(self) -> None:
        cases = [
            ({"P_binding": 0.80}, "binding", True),
            ({"P_binding": 0.70}, "binding", False),
            ({"P_activation": 0.71}, "activation", True),
            ({"P_activation": 0.70}, "activation", False),
            ({"P_inhibition": 0.71}, "inhibition", True),
            ({"P_inhibition": 0.70}, "inhibition", False),
            (
                {"P_binding": 0.80, "P_activation": 0.69},
                "binding;activation",
                False,
            ),
            (
                {"P_binding": 0.80, "P_activation": 0.71},
                "binding;activation",
                True,
            ),
            (
                {"P_binding": 0.80, "P_activation": 0.71, "P_inhibition": 0.69},
                "binding;activation;inhibition",
                False,
            ),
        ]
        for probabilities, labels, expected in cases:
            with self.subTest(probabilities=probabilities, labels=labels):
                self.assertIs(route2.event_joint_hit(probabilities, labels), expected)

    def test_all_expanded_label_rows_share_event_result(self) -> None:
        rows = pd.DataFrame(
            {
                "chemical_name": ["test", "test"],
                "AOP_ID": [1, 1],
                "Event_ID": [2, 2],
                "uniprot_id": ["P1", "P1"],
                "event_labels": ["binding;activation", "binding;activation"],
                "matched_label": ["binding", "activation"],
                "P_binding": [0.80, 0.80],
                "P_activation": [0.69, 0.69],
                "P_inhibition": [0.10, 0.10],
                "is_label_hit": [True, False],
            }
        )
        result = route2.apply_event_joint_hits(rows)
        self.assertEqual(result["is_hit"].tolist(), [False, False])
        rows["P_activation"] = 0.71
        rows["is_label_hit"] = True
        result = route2.apply_event_joint_hits(rows)
        self.assertEqual(result["is_hit"].tolist(), [True, True])


class FamilyJointHitTests(unittest.TestCase):
    @staticmethod
    def split_members() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Expansion_Member_Gene": ["A", "B"],
                "P_binding": [0.80, 0.60],
                "P_activation": [0.60, 0.80],
            }
        )

    def test_different_members_cannot_supply_different_labels(self) -> None:
        for module in (route2, family_display):
            with self.subTest(module=module.__name__):
                scored = module.score_family_members(
                    self.split_members(), "binding;activation"
                )
                self.assertFalse(scored["member_joint_hit"].any())

    def test_one_member_can_jointly_hit_and_is_representative(self) -> None:
        members = pd.concat(
            [
                self.split_members(),
                pd.DataFrame(
                    {
                        "Expansion_Member_Gene": ["C"],
                        "P_binding": [0.75],
                        "P_activation": [0.76],
                    }
                ),
            ],
            ignore_index=True,
        )
        for module in (route2, family_display):
            with self.subTest(module=module.__name__):
                scored = module.score_family_members(members, "binding;activation")
                hits = scored[scored["member_joint_hit"]]
                self.assertEqual(hits["Expansion_Member_Gene"].tolist(), ["C"])
                representative = scored.loc[scored["member_joint_score"].idxmax()]
                self.assertEqual(representative["Expansion_Member_Gene"], "C")
                self.assertAlmostEqual(float(representative["member_joint_score"]), 0.75)


if __name__ == "__main__":
    unittest.main()

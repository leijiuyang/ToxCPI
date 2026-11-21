import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aop_matching import event_joint_hit, load_mapping, load_prediction, match_chemical

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class FinalMatchingTests(unittest.TestCase):
    def mapping(self, labels="binding;activation", members="P00001;P00002", event="1"):
        return pd.DataFrame([dict(mapping_id="mapping:"+event, AOP_ID="1", Event_ID=event,
            Protein_Name="curated", Event_Labels=labels, Resolved_UniProt_ID=members)])

    def test_thresholds_and_missing_required_prediction(self):
        for task in ("binding", "activation", "inhibition"):
            for value, expected in [(0.7, False), (0.70000001, True), (0.69, False), (None, False)]:
                with self.subTest(task=task, value=value):
                    self.assertEqual(event_joint_hit({"P_"+task: value}, task), expected)
        self.assertFalse(event_joint_hit({"P_binding": 0.9}, "binding;activation"))

    def test_no_universal_binding_gate_and_both_directions(self):
        pred = pd.DataFrame([dict(uniprot_id="P00001", P_binding=0.01, P_activation=0.9, P_inhibition=0.8)]).set_index("uniprot_id")
        mapping = pd.concat([self.mapping("activation", "P00001", "1"), self.mapping("inhibition", "P00001", "2")])
        nodes, _, _ = match_chemical(mapping, pred, "test")
        self.assertEqual(nodes.is_hit.tolist(), [True, True])

    def test_cross_member_and_same_member_and_one_node(self):
        pred = pd.DataFrame([dict(uniprot_id="P00001", P_binding=0.9, P_activation=0.6),
                             dict(uniprot_id="P00002", P_binding=0.6, P_activation=0.9)]).set_index("uniprot_id")
        nodes, _, _ = match_chemical(self.mapping(), pred, "test")
        self.assertEqual(len(nodes), 1)
        self.assertFalse(nodes.iloc[0].is_hit)
        pred.loc["P00002", "P_binding"] = 0.8
        nodes, _, _ = match_chemical(self.mapping(), pred, "test")
        self.assertEqual(nodes.iloc[0].qualifying_members, "P00002")
        pred.loc["P00001", "P_activation"] = 0.8
        nodes, _, _ = match_chemical(self.mapping(), pred, "test")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes.iloc[0].qualifying_members, "P00001;P00002")

    def test_missing_member_and_shared_evidence(self):
        pred = pd.DataFrame([dict(uniprot_id="P00001", P_activation=0.9)]).set_index("uniprot_id")
        mapping = pd.concat([self.mapping("activation", event="1"), self.mapping("activation", "P00001", "2")])
        nodes, members, evidence = match_chemical(mapping, pred, "test|name")
        self.assertEqual(nodes.is_hit.tolist(), [True, True])
        self.assertEqual(json.loads(nodes.iloc[0].missing_required_predictions)[0]["uniprot_id"], "P00002")
        available = evidence[evidence.prediction_available]
        self.assertEqual(len(available), 1)
        self.assertTrue(available.iloc[0].shared_evidence)
        self.assertEqual(available.iloc[0].n_qualifying_mapping_nodes, 2)

    def test_final_input_identity_and_inline_members_only(self):
        mapping = load_mapping(ROOT / "data/AOP/human_aop_proteins_multilabel.csv")
        self.assertEqual(len(mapping), 91)
        self.assertEqual(mapping.AOP_ID.nunique(), 62)
        members = mapping.Resolved_UniProt_ID.str.split(";")
        self.assertEqual(sum(map(len, members)), 241)
        self.assertEqual(len(set(uid for group in members for uid in group)), 147)
        self.assertEqual(sum(map(lambda x: len(x)>1, members)), 14)
        self.assertIn("P31644", set(uid for group in members for uid in group))
        self.assertIn("Q16445", set(uid for group in members for uid in group))

    def test_ambiguous_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pred.csv"
            for values in ([0.8, 0.9], [0.8, 1.1]):
                pd.DataFrame(dict(chemical_name=["x", "x"], uniprot_id=["P00001", "P00001"], P_activation=values)).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_prediction(path, "x")
            pd.DataFrame(dict(chemical_name=["x"], uniprot_id=["P00001"], P_activation=[1.1])).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_prediction(path, "x")

    def test_cli_needs_only_final_table_even_with_no_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            # There are deliberately no old inventory/manual/family files here.
            import shutil
            shutil.copyfile(ROOT / "data/AOP/human_aop_proteins_multilabel.csv", work/"human_aop_proteins_multilabel.csv")
            pd.DataFrame(dict(chemical_name=["x"], slug=["x"])).to_csv(work/"chemicals.csv", index=False)
            pd.DataFrame(dict(chemical_name=["x"], uniprot_id=["P10275"], P_binding=[0.7], P_activation=[0.7], P_inhibition=[0.7])).to_csv(work/"all_human_proteome_scores_x.csv", index=False)
            subprocess.run([sys.executable, str(HERE/"05_run_aop_ppi_and_summary.py"), "--prediction-dir", str(work), "--aop-data-dir", str(work), "--chemical-list", str(work/"chemicals.csv"), "--output-dir", str(work/"out")], check=True, capture_output=True)
            nodes = pd.read_csv(work/"out/route2_event_mapping_nodes.csv")
            self.assertEqual(len(nodes), 91)
            self.assertFalse(nodes.is_hit.any())
            self.assertFalse(any("score" in col.lower() or "rank" in col.lower() for col in nodes.columns))
            self.assertEqual(len(list((work/"out").glob("*.csv"))), 5)


if __name__ == "__main__":
    unittest.main()

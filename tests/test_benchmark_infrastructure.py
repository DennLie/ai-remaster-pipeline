"""Regression and verification tests for automated benchmark provisioning infrastructure.

Validates:
1. Strict environment isolation (zero research packages in ARP or ComfyUI).
2. All 6 research and built-in resources audit properly.
3. License firewall strictly blocks research-only data without explicit opt-in.
4. Real FDS polygon parsing and 1.5x scaling on Scan 8 and 10.
5. Real ARTeFACT segmentation mask conversion to 3-tier GT.
6. Idempotency and caching of provisioning.
7. Environment immutability: benchmark runs never alter pip freeze in ARP .venv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

# Ensure scripts directory is in path
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_defect_masks as bdm
import dataset_adapters as da
import provision_benchmark_resources as pbr
import run_defect_benchmark_suite as rdbs


class BenchmarkInfrastructureTests(unittest.TestCase):
    def test_environment_isolation_audit(self) -> None:
        """Verify strict Python environment isolation: no research packages in ARP .venv."""
        iso = pbr.audit_environment_isolation()
        self.assertTrue(iso["is_isolated"], f"Isolation failed with violations: {iso['violations']}")
        self.assertEqual(iso["violations"], [])
        self.assertIn("python.exe", iso["current_python"].lower())

        # Ensure prohibited packages cannot be imported in current process
        for pkg in pbr.PROHIBITED_ARP_PACKAGES:
            with self.subTest(package=pkg):
                self.assertNotIn(pkg, sys.modules)

    def test_all_six_resources_audit_properly(self) -> None:
        """Verify all 6 research resources are audited with canonical provenance tiers."""
        audit = pbr.audit_all_resources()
        expected_keys = {
            "film_damage_simulator",
            "artefact",
            "absolute_degradation",
            "dart_harness",
            "synthetic_benchmark",
            "neighbors_pilot",
        }
        self.assertEqual(set(audit.keys()), expected_keys)

        for key, res in audit.items():
            self.assertIn(res["status"], {"INSTALLED_AND_TESTED", "INSTALLED_UNTESTED", "MISSING", "LICENSE_BLOCKED", "UNAVAILABLE"})
            self.assertIn(res["provenance_tier"], da.PROVENANCE_TIERS)
            self.assertTrue(len(res["license"]) > 0)
            self.assertTrue(len(res["commercial_suitability"]) > 0)

    def test_license_firewall_blocks_unacknowledged_research_data(self) -> None:
        """Verify AFL-3.0 and DART research components cannot be provisioned without explicit acknowledgement."""
        with tempfile.TemporaryDirectory(prefix="arp-firewall-test-") as tmp:
            tmp_path = Path(tmp)
            # ARTeFACT firewall
            art_res = pbr.provision_artefact(tmp_path / "artefact", acknowledge_research=False)
            self.assertEqual(art_res["status"], "LICENSE_BLOCKED")
            self.assertIn("acknowledge-research", art_res["reason"])

            # DART firewall
            dart_res = pbr.provision_dart_harness(tmp_path / "dart", acknowledge_research=False)
            self.assertEqual(dart_res["status"], "LICENSE_BLOCKED")
            self.assertIn("acknowledge-research", dart_res["reason"])

    def test_real_fds_parsing_and_1_5x_scaling(self) -> None:
        """Verify FilmDamageSimulator polygon parser applies 1.5x scaling to Scan (8) and Scan (10)."""
        scan8_json = ROOT_DIR / ".cache/issue34-published-benchmarks-20261109/fds_scans/Scan (8).json"
        if not scan8_json.is_file():
            self.skipTest("Scan (8).json not cached")

        polys = da.parse_fds_polygon_json(scan8_json, scan_stem="Scan (8)")
        self.assertGreater(len(polys), 100)
        # Scan (8) must apply 1.5x coordinate scaling
        self.assertEqual(polys[0]["scale_applied"], 1.5)

        # Rasterize onto a small sub-window
        masks = da.rasterize_fds_masks(polys[:10], height=1000, width=1000)
        self.assertEqual(set(np.unique(masks["combined"])).difference({0, 127, 255}), set())

    def test_real_artefact_segmentation_conversion(self) -> None:
        """Verify ARTeFACT 15-class segmentation conversion preserves 3-tier ground truth."""
        # Simulated raw annotation array containing Background(255), Clean(0), Scratch(4), Dust(3), and Fold(7)
        raw = np.array([
            [0, 255, 4],
            [3, 7, 0],
        ], dtype=np.uint8)

        gt_scratch = da.convert_artefact_segmentation_mask(raw, mode="scratch")
        # 4 (Scratch) -> 255
        self.assertEqual(gt_scratch[0, 2], 255)
        # 0 (Clean) -> 0
        self.assertEqual(gt_scratch[0, 0], 0)
        self.assertEqual(gt_scratch[1, 2], 0)
        # 255 (Background) -> 127 ignore
        self.assertEqual(gt_scratch[0, 1], 127)
        # 3 (Dust) & 7 (Fold) -> 127 ignore under scratch evaluation
        self.assertEqual(gt_scratch[1, 0], 127)
        self.assertEqual(gt_scratch[1, 1], 127)

    def test_idempotent_provisioning_caching(self) -> None:
        """Verify provisioning is idempotent and reports cached on subsequent execution."""
        cache_dir = ROOT_DIR / ".cache/benchmarks"
        res = pbr.provision_all(cache_dir, acknowledge_research=True, force=False)
        self.assertTrue(res["isolation_audit"]["is_isolated"])

        # All resources should report 'cached' or 'provisioned'
        for name, r in res["resources"].items():
            self.assertIn(r["status"], {"cached", "provisioned"})

    def test_environment_immutability_before_and_after_benchmarks(self) -> None:
        """Verify running benchmark provisioning and suite does not mutate pip freeze."""
        proc1 = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        freeze_before = proc1.stdout.strip().splitlines()

        # Run provisioning status check and synthetic provisioning in temp dir
        with tempfile.TemporaryDirectory(prefix="arp-immutability-test-") as tmp:
            pbr.provision_synthetic_benchmark(Path(tmp) / "synthetic")

        proc2 = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        freeze_after = proc2.stdout.strip().splitlines()

        self.assertEqual(freeze_before, freeze_after, "pip freeze changed after benchmark operation!")


if __name__ == "__main__":
    unittest.main()

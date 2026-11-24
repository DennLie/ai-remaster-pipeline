"""Comprehensive unit tests for Selective Re-Render and Multi-Worker Batch Dispatch.

Covers:
1. Selective Re-Render Isolation: targeting fail shots while preserving pass/warn shots.
2. Multi-Worker Shard Partitioning (N=2, N=4) with zero dropped shots and zero overlaps.
3. Shard manifest generation with valid headers and Color Bible snapshot injection.
4. Merge shard manifests restoring deterministic shot_index order.
5. Order-independent execution determinism (reverse order vs forward order).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import cv2
import numpy as np

from ai_remaster_gui.color_bible_db import (
    Film,
    QCEvent,
    ShotRecord,
    SQLiteColorBibleRepository,
    apply_migrations,
    get_connection,
    record_qc_event,
)
from ai_remaster_gui.commercial_policy import CommercialPolicyError
from ai_remaster_gui.manifests import read_manifest_details, write_manifest_details
from ai_remaster_gui.paths import ROOT
from scripts.benchmark_colorizers import (
    main as benchmark_main,
    run_benchmark,
)
from scripts.parallel_colorize_dispatch import (
    dispatch_local_workers,
    generate_shard_manifests,
    main as dispatch_main,
    merge_shard_manifests,
    partition_manifest_rows,
)
from scripts.rerender_failed_shots import (
    extract_shot_identifier,
    find_failed_shots,
    main as rerender_main,
    rerender_failed_shots,
)


def create_sample_manifest(
    path: Path,
    source_video: str = "media/input_film.mp4",
    num_shots: int = 4,
    statuses: list[str] | None = None,
) -> list[dict[str, str]]:
    """Helper to create a deterministic sample shot manifest."""
    fieldnames = [
        "shot_id",
        "shot_index",
        "start_frame",
        "end_frame",
        "color_reference",
        "qc_status",
        "enabled",
    ]
    rows: list[dict[str, str]] = []
    default_statuses = ["pass", "pass", "warn", "fail"]

    for i in range(num_shots):
        status = statuses[i] if statuses and i < len(statuses) else default_statuses[i % len(default_statuses)]
        row = {
            "shot_id": f"shot_{i + 1:03d}",
            "shot_index": str(i),
            "start_frame": str(i * 100),
            "end_frame": str((i + 1) * 100),
            "color_reference": f"refs/ref_{i + 1:03d}.png",
            "qc_status": status,
            "enabled": "true",
        }
        rows.append(row)

    write_manifest_details(path, source_video, fieldnames, rows)
    return rows


class TestSelectiveReRender(unittest.TestCase):
    """Test selective re-rendering of failed QC shots and preservation of passing shots."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)
        self.manifest_path = self.work_dir / "shots_manifest.csv"
        # 4 shots: 2 pass, 1 warn, 1 fail
        self.rows = create_sample_manifest(
            self.manifest_path,
            num_shots=4,
            statuses=["pass", "pass", "warn", "fail"],
        )

    def tearDown(self) -> None:
        try:
            self.tmp_dir.cleanup()
        except OSError:
            pass

    def test_rerender_failed_shots_dry_run_identifies_only_fail_shot(self) -> None:
        """Verify dry-run targets ONLY the fail shot while strictly preserving others."""
        res = rerender_failed_shots(self.manifest_path, dry_run=True)

        self.assertEqual(res["total_shots"], 4)
        self.assertEqual(res["failed_shot_ids"], ["shot_004"])
        self.assertEqual(res["rerendered_shot_ids"], [])
        self.assertEqual(res["preserved_shot_ids"], ["shot_001", "shot_002", "shot_003"])

        # Master manifest must remain unmodified
        _src, _fields, current_rows = read_manifest_details(self.manifest_path)
        self.assertEqual([r["qc_status"] for r in current_rows], ["pass", "pass", "warn", "fail"])

    def test_rerender_failed_shots_with_mock_executor(self) -> None:
        """Verify re-rendering only invokes executor on fail shot and accurately updates manifest."""
        executed_shots: list[str] = []

        def mock_executor(sub_manifest_path: Path) -> None:
            src, fields, sub_rows = read_manifest_details(sub_manifest_path)
            if "colorized_video" not in fields:
                fields.append("colorized_video")
            for r in sub_rows:
                executed_shots.append(r["shot_id"])
                # Simulate successful re-render: fix status and record output
                r["qc_status"] = "pass"
                r["colorized_video"] = f"output/{r['shot_id']}_rerendered.mp4"
            write_manifest_details(sub_manifest_path, src, fields, sub_rows)

        res = rerender_failed_shots(self.manifest_path, execute_fn=mock_executor)

        # 1. Only the 1 fail shot was passed to the executor
        self.assertEqual(executed_shots, ["shot_004"])
        self.assertEqual(res["failed_shot_ids"], ["shot_004"])
        self.assertEqual(res["rerendered_shot_ids"], ["shot_004"])

        # 2. The 2 pass shots and 1 warn shot were strictly preserved and never executed
        self.assertNotIn("shot_001", executed_shots)
        self.assertNotIn("shot_002", executed_shots)
        self.assertNotIn("shot_003", executed_shots)
        self.assertEqual(res["preserved_shot_ids"], ["shot_001", "shot_002", "shot_003"])

        # 3. Manifest is updated accurately
        _src, _fields, final_rows = read_manifest_details(self.manifest_path)
        self.assertEqual(len(final_rows), 4)

        # Pass shots preserved untouched
        self.assertEqual(final_rows[0]["shot_id"], "shot_001")
        self.assertEqual(final_rows[0]["qc_status"], "pass")
        self.assertEqual(final_rows[0].get("colorized_video", ""), "")

        self.assertEqual(final_rows[1]["shot_id"], "shot_002")
        self.assertEqual(final_rows[1]["qc_status"], "pass")
        self.assertEqual(final_rows[1].get("colorized_video", ""), "")

        # Warn shot preserved untouched
        self.assertEqual(final_rows[2]["shot_id"], "shot_003")
        self.assertEqual(final_rows[2]["qc_status"], "warn")
        self.assertEqual(final_rows[2].get("colorized_video", ""), "")

        # Fail shot successfully updated
        self.assertEqual(final_rows[3]["shot_id"], "shot_004")
        self.assertEqual(final_rows[3]["qc_status"], "pass")
        self.assertEqual(final_rows[3]["colorized_video"], "output/shot_004_rerendered.mp4")

    def test_rerender_failed_shots_with_color_bible_db(self) -> None:
        """Verify Color Bible DB qc_events can trigger re-renders even if manifest is unflagged."""
        # Create manifest where manifest row has empty or 'pass' status
        clean_manifest = self.work_dir / "db_manifest.csv"
        create_sample_manifest(clean_manifest, num_shots=3, statuses=["pass", "pass", "pass"])

        # Set up SQLite DB with a failed QCEvent for shot_002
        db_path = self.work_dir / "color_bible.db"
        repo = SQLiteColorBibleRepository(db_path)
        repo.init_db()

        film = Film(film_id="film_rerender_test", title="Test Movie")
        repo.upsert_film(film)

        shot = ShotRecord(
            shot_id="shot_002",
            film_id="film_rerender_test",
            shot_index=1,
            start_frame=100,
            end_frame=200,
            start_time=4.16,
            end_time=8.33,
        )
        repo.upsert_shot(shot)

        ev_fail = QCEvent(
            event_id="ev_fail_test_2",
            film_id="film_rerender_test",
            shot_id="shot_002",
            shot_index=1,
            qc_status="fail",
            failure_codes=["COLOR_BLEED"],
            metrics={"delta_e": 12.5},
        )
        repo.record_qc_event(ev_fail)
        repo.close()

        res = rerender_failed_shots(
            clean_manifest,
            color_bible_db=db_path,
            film_id="film_rerender_test",
            dry_run=True,
        )

        self.assertEqual(res["total_shots"], 3)
        self.assertEqual(res["failed_shot_ids"], ["shot_002"])
        self.assertEqual(res["preserved_shot_ids"], ["shot_001", "shot_003"])

        # Also test with film_id=None to verify find_failed_shots cleanly handles None film_id
        res_none_film = rerender_failed_shots(
            clean_manifest,
            color_bible_db=db_path,
            film_id=None,
            dry_run=True,
        )
        self.assertEqual(res_none_film["failed_shot_ids"], ["shot_002"])

        _src, _fields, clean_rows = read_manifest_details(clean_manifest)
        failed_ids, preserved_ids = find_failed_shots(clean_rows, color_bible_db=db_path, film_id=None)
        self.assertEqual(failed_ids, ["shot_002"])
        self.assertEqual(preserved_ids, ["shot_001", "shot_003"])

    def test_rerender_no_failures_does_not_invoke_executor(self) -> None:
        """Verify when zero shots fail, executor is never called."""
        all_pass_manifest = self.work_dir / "all_pass.csv"
        create_sample_manifest(all_pass_manifest, num_shots=3, statuses=["pass", "pass", "warn"])

        mock_called = False

        def mock_executor(sub_path: Path) -> None:
            nonlocal mock_called
            mock_called = True

        res = rerender_failed_shots(all_pass_manifest, execute_fn=mock_executor)
        self.assertFalse(mock_called)
        self.assertEqual(res["total_shots"], 3)
        self.assertEqual(res["failed_shot_ids"], [])
        self.assertEqual(res["rerendered_shot_ids"], [])
        self.assertEqual(res["preserved_shot_ids"], ["shot_001", "shot_002", "shot_003"])


class TestMultiWorkerPartitionAndDeterminism(unittest.TestCase):
    """Test Multi-Worker Shard Partitioning, RunPod Shard Generation, and Determinism."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_partition_manifest_rows_coverage_and_disjointness_n2_and_n4(self) -> None:
        """Verify partition_manifest_rows covers 100% of rows with zero overlaps and zero dropped shots."""
        # Test across multiple row counts and both N=2 and N=4
        for num_shots in [4, 8, 10, 15]:
            rows = [{"shot_id": f"shot_{i}", "shot_index": str(i)} for i in range(num_shots)]

            for worker_count in [2, 4]:
                for strategy in ["interleaved", "contiguous"]:
                    all_partitioned: list[dict[str, str]] = []
                    seen_shot_ids: set[str] = set()

                    for w in range(worker_count):
                        shard_rows = partition_manifest_rows(
                            rows, worker_index=w, worker_count=worker_count, strategy=strategy
                        )

                        # Check zero overlaps
                        shard_ids = [r["shot_id"] for r in shard_rows]
                        self.assertTrue(
                            seen_shot_ids.isdisjoint(shard_ids),
                            f"Overlapping shots found in worker {w} (strategy={strategy}, N={worker_count})",
                        )
                        seen_shot_ids.update(shard_ids)
                        all_partitioned.extend(shard_rows)

                    # Check zero dropped shots and 100% coverage
                    self.assertEqual(
                        len(all_partitioned),
                        num_shots,
                        f"Dropped or extra shots for N={worker_count}, strategy={strategy}",
                    )
                    self.assertEqual(
                        len(seen_shot_ids),
                        num_shots,
                        f"Incomplete coverage for N={worker_count}, strategy={strategy}",
                    )
                    # All original shot IDs are present
                    expected_ids = {f"shot_{i}" for i in range(num_shots)}
                    self.assertEqual(seen_shot_ids, expected_ids)

    def test_generate_shard_manifests_headers_and_content(self) -> None:
        """Verify generate_shard_manifests creates valid CSV files with proper headers (# source_video=...)."""
        manifest_path = self.work_dir / "master_manifest.csv"
        source_vid = "media/restoration_master_1080p.mp4"
        create_sample_manifest(manifest_path, source_video=source_vid, num_shots=6)

        shard_paths = generate_shard_manifests(
            manifest_path=manifest_path,
            worker_count=3,
            output_dir=self.work_dir / "shards",
            strategy="interleaved",
        )

        self.assertEqual(len(shard_paths), 3)

        for i, shard_path in enumerate(shard_paths):
            self.assertTrue(shard_path.exists())
            self.assertEqual(shard_path.name, f"master_manifest_worker_{i}_of_3.csv")

            # Check valid # source_video=... header
            src, fields, rows = read_manifest_details(shard_path)
            self.assertEqual(src, source_vid)
            self.assertIn("shot_id", fields)
            self.assertIn("shot_index", fields)
            # 6 shots / 3 workers = 2 shots each
            self.assertEqual(len(rows), 2)

    def test_generate_shard_manifests_color_bible_version_injection(self) -> None:
        """Verify Color Bible snapshot version/hash is frozen into all worker shard manifests."""
        manifest_path = self.work_dir / "snapshot_manifest.csv"
        create_sample_manifest(manifest_path, num_shots=4)

        snapshot_hash = "cb_v2_f8e3a2b1"
        shard_paths = generate_shard_manifests(
            manifest_path=manifest_path,
            worker_count=2,
            output_dir=self.work_dir / "snapshot_shards",
            strategy="interleaved",
            color_bible_version=snapshot_hash,
        )

        for shard_path in shard_paths:
            _src, fields, rows = read_manifest_details(shard_path)
            self.assertIn("color_bible_version", fields)
            for row in rows:
                self.assertEqual(row.get("color_bible_version"), snapshot_hash)

    def test_merge_shard_manifests_restores_sorted_sequence(self) -> None:
        """Verify merge_shard_manifests restores the exact original sorted shot sequence."""
        manifest_path = self.work_dir / "original_manifest.csv"
        original_rows = create_sample_manifest(manifest_path, num_shots=8)

        # Generate 4 interleaved shards
        shard_paths = generate_shard_manifests(
            manifest_path=manifest_path,
            worker_count=4,
            output_dir=self.work_dir / "merge_shards",
            strategy="interleaved",
        )

        merged_path = self.work_dir / "restored_manifest.csv"
        merge_shard_manifests(shard_paths, merged_path)

        _src, _fields, merged_rows = read_manifest_details(merged_path)

        self.assertEqual(len(merged_rows), len(original_rows))
        for orig, merged in zip(original_rows, merged_rows):
            self.assertEqual(orig["shot_id"], merged["shot_id"])
            self.assertEqual(orig["shot_index"], merged["shot_index"])
            self.assertEqual(orig["qc_status"], merged["qc_status"])

    def test_order_independence_execution_determinism(self) -> None:
        """Verify executing shard 1 before shard 0 (or reverse order N-1...0) yields identical deterministic results."""
        manifest_path = self.work_dir / "determinism_manifest.csv"
        create_sample_manifest(manifest_path, num_shots=6)

        shard_paths = generate_shard_manifests(
            manifest_path=manifest_path,
            worker_count=3,
            output_dir=self.work_dir / "determinism_shards",
            strategy="interleaved",
        )

        # Simulate execution on each shard (e.g. adding a deterministic computed field)
        def process_shard(path: Path, worker_id: int) -> None:
            src, fields, rows = read_manifest_details(path)
            if "processed_by" not in fields:
                fields.append("processed_by")
            for r in rows:
                r["processed_by"] = f"worker_{worker_id}"
            write_manifest_details(path, src, fields, rows)

        # Clone shards to simulate reverse-order processing
        rev_shard_paths: list[Path] = []
        rev_dir = self.work_dir / "reverse_shards"
        rev_dir.mkdir(parents=True, exist_ok=True)
        for p in shard_paths:
            rev_p = rev_dir / p.name
            rev_p.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            rev_shard_paths.append(rev_p)

        # 1. Forward order execution: 0 -> 1 -> 2
        for i, p in enumerate(shard_paths):
            process_shard(p, i)
        forward_merged = self.work_dir / "forward_merged.csv"
        merge_shard_manifests(shard_paths, forward_merged)

        # 2. Reverse order execution: 2 -> 1 -> 0
        for i in reversed(range(len(rev_shard_paths))):
            process_shard(rev_shard_paths[i], i)
        reverse_merged = self.work_dir / "reverse_merged.csv"
        # Merge passed in reverse order
        merge_shard_manifests(list(reversed(rev_shard_paths)), reverse_merged)

        # Verify forward and reverse merged outputs are byte-for-byte identical
        forward_text = forward_merged.read_text(encoding="utf-8").strip()
        reverse_text = reverse_merged.read_text(encoding="utf-8").strip()
        self.assertEqual(forward_text, reverse_text)

    def test_dispatch_local_workers_with_mock_executor(self) -> None:
        """Verify dispatch_local_workers coordinates concurrent execution and merges shards."""
        manifest_path = self.work_dir / "dispatch_manifest.csv"
        create_sample_manifest(manifest_path, num_shots=4)

        executed_shards: list[str] = []

        def mock_worker(shard_p: Path) -> None:
            executed_shards.append(shard_p.name)
            src, fields, rows = read_manifest_details(shard_p)
            for r in rows:
                r["rendered"] = "true"
            if "rendered" not in fields:
                fields.append("rendered")
            write_manifest_details(shard_p, src, fields, rows)

        res = dispatch_local_workers(manifest_path, worker_count=2, execute_fn=mock_worker)

        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["worker_count"], 2)
        self.assertEqual(len(executed_shards), 2)

        merged_p = Path(res["merged_manifest"])
        self.assertTrue(merged_p.exists())
        _src, fields, merged_rows = read_manifest_details(merged_p)
        self.assertIn("rendered", fields)
        self.assertEqual(len(merged_rows), 4)
        for r in merged_rows:
            self.assertEqual(r.get("rendered"), "true")

    def test_partition_error_handling(self) -> None:
        """Verify invalid worker parameters and strategies raise clear ValueErrors."""
        rows = [{"shot_id": "shot_1"}]
        # worker_count < 1
        with self.assertRaises(ValueError):
            partition_manifest_rows(rows, worker_index=0, worker_count=0)

        # worker_index out of bounds
        with self.assertRaises(ValueError):
            partition_manifest_rows(rows, worker_index=2, worker_count=2)

        with self.assertRaises(ValueError):
            partition_manifest_rows(rows, worker_index=-1, worker_count=2)

        # unknown strategy
        with self.assertRaises(ValueError):
            partition_manifest_rows(rows, worker_index=0, worker_count=2, strategy="random_shuffle")


class TestCLIEntryPoints(unittest.TestCase):
    """Test CLI entry points for rerender_failed_shots and parallel_colorize_dispatch."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)
        self.manifest_path = self.work_dir / "cli_test_manifest.csv"
        create_sample_manifest(self.manifest_path, num_shots=4, statuses=["pass", "warn", "fail", "pass"])

    def tearDown(self) -> None:
        try:
            self.tmp_dir.cleanup()
        except OSError:
            pass

    def test_rerender_cli_dry_run(self) -> None:
        """Verify rerender_failed_shots CLI executes dry run successfully."""
        exit_code = rerender_main(["--manifest", str(self.manifest_path), "--dry-run"])
        self.assertEqual(exit_code, 0)

    def test_parallel_dispatch_cli_generate_shards(self) -> None:
        """Verify parallel_colorize_dispatch CLI generates all shards."""
        shards_dir = self.work_dir / "cli_shards"
        exit_code = dispatch_main([
            "--manifest", str(self.manifest_path),
            "--worker-count", "2",
            "--output-dir", str(shards_dir),
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue((shards_dir / "cli_test_manifest_worker_0_of_2.csv").exists())
        self.assertTrue((shards_dir / "cli_test_manifest_worker_1_of_2.csv").exists())

    def test_parallel_dispatch_cli_single_runpod_shard(self) -> None:
        """Verify parallel_colorize_dispatch CLI generates a single RunPod shard."""
        runpod_dir = self.work_dir / "runpod_shards"
        exit_code = dispatch_main([
            "--manifest", str(self.manifest_path),
            "--worker-index", "1",
            "--worker-count", "3",
            "--output-dir", str(runpod_dir),
            "--color-bible-snapshot", "snap_runpod_123",
        ])
        self.assertEqual(exit_code, 0)
        shard_path = runpod_dir / "cli_test_manifest_worker_1_of_3.csv"
        self.assertTrue(shard_path.exists())
        _src, fields, rows = read_manifest_details(shard_path)
        self.assertIn("color_bible_version", fields)
        for r in rows:
            self.assertEqual(r.get("color_bible_version"), "snap_runpod_123")

    def test_parallel_dispatch_cli_merge_manifests(self) -> None:
        """Verify parallel_colorize_dispatch CLI merges shard manifests."""
        shards = generate_shard_manifests(self.manifest_path, worker_count=2, output_dir=self.work_dir)
        merged_path = self.work_dir / "cli_merged.csv"
        exit_code = dispatch_main([
            "--merge-manifests", str(shards[0]), str(shards[1]),
            "--output", str(merged_path),
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(merged_path.exists())
        _src, _f, rows = read_manifest_details(merged_path)
        self.assertEqual(len(rows), 4)


class TestBenchmarkColorizers(unittest.TestCase):
    """Test suite for benchmark_colorizers: compliance safeguards, metric computations, and reports."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)
        self.frames_dir = self.work_dir / "input_frames"
        self.gt_dir = self.work_dir / "gt_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)

        # Generate 4 deterministic sample frames (64x64)
        for i in range(4):
            # Input frames (grayscale)
            gray = np.full((64, 64), 100 + i * 20, dtype=np.uint8)
            cv2.imwrite(str(self.frames_dir / f"frame_{i:03d}.png"), gray)

            # Ground-truth frames (color BGR for cv2.imwrite)
            b = np.full((64, 64), 50 + i * 10, dtype=np.uint8)
            g = np.full((64, 64), 120 + i * 15, dtype=np.uint8)
            r = np.full((64, 64), 180 - i * 10, dtype=np.uint8)
            bgr = np.stack([b, g, r], axis=-1)
            cv2.imwrite(str(self.gt_dir / f"frame_{i:03d}.png"), bgr)

    def tearDown(self) -> None:
        try:
            self.tmp_dir.cleanup()
        except OSError:
            pass

    def test_fails_closed_without_non_commercial_authorization(self) -> None:
        """Verify benchmark fails closed immediately unless allow_non_commercial=True."""
        out_dir = self.work_dir / "reports"
        with self.assertRaises(CommercialPolicyError):
            run_benchmark(
                frames_dir=self.frames_dir,
                ground_truth_dir=self.gt_dir,
                output_dir=out_dir,
                allow_non_commercial=False,
                mock_cmnet2=True,
            )

        # CLI entrypoint fails closed when --allow-non-commercial-benchmark is missing
        with self.assertRaises(CommercialPolicyError):
            benchmark_main([
                "--frames-dir", str(self.frames_dir),
                "--ground-truth-dir", str(self.gt_dir),
                "--output-dir", str(out_dir),
                "--mock-cmnet2",
            ])

    def test_rejects_output_in_commercial_master_folder(self) -> None:
        """Verify benchmark strictly prohibits writing inside ROOT / 'output' commercial master paths."""
        commercial_out = ROOT / "output"
        with self.assertRaises(CommercialPolicyError):
            run_benchmark(
                frames_dir=self.frames_dir,
                ground_truth_dir=self.gt_dir,
                output_dir=commercial_out,
                allow_non_commercial=True,
                mock_cmnet2=True,
            )

        commercial_sub = ROOT / "output" / "commercial_deliverable"
        with self.assertRaises(CommercialPolicyError):
            run_benchmark(
                frames_dir=self.frames_dir,
                ground_truth_dir=self.gt_dir,
                output_dir=commercial_sub,
                allow_non_commercial=True,
                mock_cmnet2=True,
            )

    def test_end_to_end_mock_cmnet2_benchmark_cpu(self) -> None:
        """Verify end-to-end benchmark execution on CPU with mock_cmnet2=True."""
        out_dir = self.work_dir / "benchmark_run"
        res = run_benchmark(
            frames_dir=self.frames_dir,
            ground_truth_dir=self.gt_dir,
            output_dir=out_dir,
            allow_non_commercial=True,
            mock_cmnet2=True,
            stride=1,
        )

        # 1. Structured JSON and Markdown reports are produced
        report_json_path = out_dir / "report.json"
        report_md_path = out_dir / "report.md"
        self.assertTrue(report_json_path.exists())
        self.assertTrue(report_md_path.exists())

        with open(report_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        md_text = report_md_path.read_text(encoding="utf-8")
        self.assertIn("# Colorizer Benchmark Report", md_text)
        self.assertIn("osi_masked_lab", md_text)
        self.assertIn("CMNET2", md_text)

        # 2. PSNR, mean CIEDE2000, p90 CIEDE2000, and CDC are computed and present
        for method_key in ("osi_masked_lab", "cmnet2"):
            metrics = res["methods"][method_key]
            self.assertIn("psnr_db", metrics)
            self.assertIn("mean_psnr_db", metrics)
            self.assertIn("mean_ciede2000", metrics)
            self.assertIn("p90_ciede2000", metrics)
            self.assertIn("cdc", metrics)

            self.assertIsInstance(metrics["psnr_db"], (int, float))
            self.assertGreater(metrics["psnr_db"], 0.0)
            self.assertIsInstance(metrics["mean_ciede2000"], (int, float))
            self.assertGreaterEqual(metrics["mean_ciede2000"], 0.0)
            self.assertIsInstance(metrics["p90_ciede2000"], (int, float))
            self.assertGreaterEqual(metrics["p90_ciede2000"], 0.0)
            self.assertIsInstance(metrics["cdc"], (int, float))
            self.assertGreaterEqual(metrics["cdc"], 0.0)
            self.assertLessEqual(metrics["cdc"], 1.0)

        # 3. Compliance record is emitted with NON_COMMERCIAL_BENCHMARK
        compliance_path = out_dir / "compliance_report.json"
        self.assertTrue(compliance_path.exists())
        with open(compliance_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)

        found_non_commercial_record = False
        for run in comp_data.get("runs", []):
            for model in run.get("models", []):
                if model.get("model_id") == "CMNET2" and "NON_COMMERCIAL_BENCHMARK" in str(model.get("reason", "")):
                    found_non_commercial_record = True
            if run.get("metadata", {}).get("benchmark_type") == "NON_COMMERCIAL_BENCHMARK":
                found_non_commercial_record = True
        self.assertTrue(found_non_commercial_record, "Expected compliance record with NON_COMMERCIAL_BENCHMARK")

    def test_benchmark_cli_invocation_with_mock(self) -> None:
        """Verify CLI entrypoint succeeds with mock_cmnet2 and valid flags."""
        cli_out = self.work_dir / "cli_report"
        exit_code = benchmark_main([
            "--frames-dir", str(self.frames_dir),
            "--ground-truth-dir", str(self.gt_dir),
            "--output-dir", str(cli_out),
            "--allow-non-commercial-benchmark",
            "--mock-cmnet2",
            "--stride", "2",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue((cli_out / "report.json").exists())
        self.assertTrue((cli_out / "report.md").exists())


if __name__ == "__main__":
    unittest.main()

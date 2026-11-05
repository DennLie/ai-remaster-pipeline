import concurrent.futures
import threading
import unittest
from unittest.mock import patch

from ai_remaster_gui import denoise_review


class ReadinessSnapshotTests(unittest.TestCase):
    def test_reuses_per_app_and_stage_but_checks_again_on_next_request(self):
        first, second = object(), object()

        @denoise_review.readiness_snapshot
        def response():
            for app, stage in ((first, 'denoise'), (first, 'enhancement'), (second, 'denoise')):
                self.assertEqual(denoise_review.output_ready(app, stage), denoise_review.output_ready(app, stage))

        with patch.object(denoise_review, 'selected_output') as check:
            response()
            self.assertEqual(check.call_count, 3)
            check.side_effect = ValueError('Provenance changed')
            response()
            self.assertEqual(check.call_count, 6)
            self.assertFalse(denoise_review.output_ready(first, 'denoise'))
            self.assertEqual(check.call_count, 7)

    def test_nested_snapshot_and_exception_restore_the_previous_scope(self):
        app = object()

        @denoise_review.readiness_snapshot
        def failing():
            self.assertFalse(denoise_review.output_ready(app, 'denoise'))
            raise RuntimeError('Failed state request')

        @denoise_review.readiness_snapshot
        def outer():
            self.assertTrue(denoise_review.output_ready(app, 'denoise'))
            check.side_effect = ValueError('Changed')
            with self.assertRaises(RuntimeError):
                failing()
            self.assertTrue(denoise_review.output_ready(app, 'denoise'))

        with patch.object(denoise_review, 'selected_output') as check:
            outer()
            self.assertFalse(denoise_review.output_ready(app, 'denoise'))
            self.assertEqual(check.call_count, 3)

    def test_concurrent_requests_do_not_share_readiness(self):
        barrier = threading.Barrier(2)
        app = object()

        @denoise_review.readiness_snapshot
        def response():
            ready = denoise_review.output_ready(app, 'denoise')
            barrier.wait(timeout=5)
            return ready, denoise_review.output_ready(app, 'denoise')

        with patch.object(denoise_review, 'selected_output') as check, concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(response) for _ in range(2)]
            self.assertEqual([f.result() for f in futures], [(True, True), (True, True)])
            self.assertEqual(check.call_count, 2)

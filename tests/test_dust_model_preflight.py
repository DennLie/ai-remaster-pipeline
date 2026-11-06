"""Read-only prerequisites: no prompt, download, server start or model inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import dust
import dust_model_preflight as preflight


class ModelPreflightTests(unittest.TestCase):
    def weights(self, root):
        folder=root/'custom_nodes/ComfyUI_ProPainter_Nodes/weights'
        folder.mkdir(parents=True)
        for name in preflight.PROPAINTER_WEIGHTS:
            (folder/name).write_bytes(b'fixture; not actual model weights')

    def test_missing_weights_block_before_contacting_comfy(self):
        with tempfile.TemporaryDirectory() as td,patch.object(preflight,'http_json') as http:
            report=preflight.inspect_propainter(Path(td),'http://localhost:8188')
        self.assertFalse(report['ready']);self.assertIn('No automatic download',report['blockers'][0]);http.assert_not_called()

    def test_busy_queue_blocks_before_node_query_or_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.weights(root)
            with patch.object(preflight.subprocess,'check_output',side_effect=['a'*40,'']), \
                 patch.object(preflight,'http_json',return_value={'queue_running':[[1]],'queue_pending':[]}) as http:
                report=preflight.inspect_propainter(root,'http://localhost:8188')
            self.assertFalse(report['ready']);self.assertEqual(http.call_count,1)
            self.assertEqual(http.call_args.args[0],'GET')

    def test_live_schema_and_file_inventory_are_read_only(self):
        fields=('image','mask','width','height','mask_dilates','flow_mask_dilates','ref_stride','neighbor_length','subvideo_length','raft_iter','fp16')
        schema={name:{} for name in preflight.PROPAINTER_NODES}
        schema['ProPainterInpaint']={'input':{'required':{name:[] for name in fields}}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.weights(root)
            with patch.object(preflight.subprocess,'check_output',side_effect=['a'*40,'']), \
                 patch.object(preflight,'http_json',side_effect=[{'queue_running':[],'queue_pending':[]},schema]) as http:
                report=preflight.inspect_propainter(root,'http://localhost:8188')
            self.assertTrue(report['ready']);self.assertEqual(len(report['weights']),3)
            self.assertTrue(all(len(w['sha256'])==64 for w in report['weights']))
            self.assertTrue(all(call.args[0]=='GET' for call in http.call_args_list))

    def test_unknown_node_contract_is_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.weights(root)
            with patch.object(preflight.subprocess,'check_output',side_effect=['a'*40,'']), \
                 patch.object(preflight,'http_json',side_effect=[{'queue_running':[],'queue_pending':[]},{name:{} for name in preflight.PROPAINTER_NODES}]):
                report=preflight.inspect_propainter(root,'http://localhost:8188')
            self.assertFalse(report['ready']);self.assertIn('schema',report['blockers'][0])

    def test_dirty_dart_checkout_never_probes_or_executes_inference(self):
        with tempfile.TemporaryDirectory() as td,patch.object(preflight,'validate_checkout'), \
             patch.object(preflight.subprocess,'check_output',return_value=' M scripts/inference.py') as process:
            report=preflight.inspect_dart(Path(td),Path(td)/'python.exe')
        self.assertFalse(report['ready']);self.assertEqual(process.call_count,1)
        self.assertIn('modifications',report['blockers'][0])

    def test_require_ready_cannot_ignore_a_blocker(self):
        with self.assertRaisesRegex(ValueError,'offline'):
            preflight.require_ready({'ready':False,'backend':'propainter','blockers':['offline']})


if __name__=='__main__':unittest.main()

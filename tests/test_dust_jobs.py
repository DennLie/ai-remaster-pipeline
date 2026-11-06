"""Dust job failure injection with real FFV1 files; no GPU/server required."""
import json
import csv
import tempfile
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
from ai_remaster_gui import dust, server
from ai_remaster_gui.dust_jobs import candidate_lock
import cleanup_video


class AnalysisPublicationTests(unittest.TestCase):
    def test_publication_failure_retains_complete_analysis_and_retry_does_not_detect(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            rename=Path.rename
            stages=[]
            def deny(path,destination):
                if Path(destination)==dust.folder_for(source,{}):
                    stages.append(path)
                    raise PermissionError(13,'injected locked result folder')
                return rename(path,destination)
            with patch.object(Path,'rename',deny),self.assertRaises(OSError):
                dust.run(source,{},'mask')
            stage=stages[0]
            self.assertTrue((stage/'report.json').is_file(),'Publication failure deleted completed analysis')
            report=dust.manifest(stage,source,{},'mask')
            with self.assertRaises(FileNotFoundError):dust.saved_mask(source,{})
            self.assertEqual(dust.matrix_state(SimpleNamespace(stabilized_source_for_downstream=lambda:str(source),running_stage_key=''))['variants'],[])
            # Corruption and absent reports must fail closed, without regenerating masks.
            mask=stage/'effective_mask.mkv';original=mask.read_bytes();mask.write_bytes(b'changed')
            with patch.object(dust,'masks_for',side_effect=AssertionError('must not detect again')),self.assertRaisesRegex(ValueError,'Pending Dust'):
                dust.run(source,{},'mask')
            mask.write_bytes(original)
            with patch.object(dust,'masks_for',side_effect=AssertionError('must not detect again')):
                dust.run(source,{},'mask')
            target=dust.folder_for(source,{})
            self.assertEqual(dust.manifest(target,source,{},'mask'),report)
            self.assertFalse((target/'repaired.mkv').exists())

    def test_incomplete_pending_analysis_is_preserved_and_never_recomputed(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv';source.write_bytes(b'fixture')
            target=dust.folder_for(source,{})
            pending=target.parent/f'.{target.name}.analysis-fixture';pending.mkdir(parents=True)
            marker=pending/'partial';marker.write_bytes(b'keep')
            with patch.object(dust,'masks_for',side_effect=AssertionError('must not detect')),self.assertRaisesRegex(ValueError,'incomplete'):
                dust.run(source,{},'mask')
            self.assertEqual(marker.read_bytes(),b'keep');self.assertFalse(target.exists())

    def test_transient_publication_lock_is_retried(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            target=dust.folder_for(source,{});rename=Path.rename;attempts=[]
            def transient(path,destination):
                if Path(destination)==target:
                    attempts.append(path)
                    if len(attempts)<3:raise PermissionError(13,'transient lock')
                return rename(path,destination)
            with patch.object(Path,'rename',transient),patch.object(dust.time,'sleep'):
                dust.run(source,{},'mask')
            self.assertEqual(len(attempts),3)
            dust.manifest(target,source,{},'mask')

    def test_pending_dart_full_frames_are_validated_before_publication(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv';source.write_bytes(b'fixture')
            values=dust.options({'detector':'dart','repair':'dart'})
            target=dust.folder_for(source,values);stage=target.parent/f'.{target.name}.analysis-fixture';stage.mkdir(parents=True)
            hashes={}
            for name in ('input.mkv','mask.mkv','effective_mask.mkv','measurements.csv','dart_full.mkv'):
                (stage/name).write_bytes(b'fixture');hashes[name]=dust.digest(stage/name)
            report={'identity':dust.identity(source,values),'source_sha256':dust.digest(source),'repair_complete':False,'hashes':hashes}
            dust.publish_report(stage,report)
            (stage/'dart_full.mkv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Pending Dust'):dust.finish_analysis(stage,target,source,values)
            self.assertFalse(target.exists())
            (stage/'dart_full.mkv').write_bytes(b'fixture')
            dust.finish_analysis(stage,target,source,values)
            self.assertEqual(dust.manifest(target,source,values,'mask'),report)

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_real_windows_file_lock_cannot_delete_finished_analysis(self):
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            publish=dust.publish_report;handles=[];stages=[]
            def hold(folder,report):
                publish(folder,report);stages.append(folder)
                handle=kernel.CreateFileW(str(folder/'input.mkv'),0x80000000,3,None,3,0,None)
                if handle==wintypes.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
                handles.append(handle)
            try:
                with patch.object(dust,'publish_report',side_effect=hold),self.assertRaises(OSError):
                    dust.run(source,{},'mask')
                self.assertTrue((stages[0]/'report.json').is_file(),'Locked publication triggered destructive cleanup')
                self.assertTrue((stages[0]/'effective_mask.mkv').is_file())
            finally:
                for handle in handles:kernel.CloseHandle(handle)
            with patch.object(dust,'masks_for',side_effect=AssertionError('must reuse finished analysis')):
                dust.run(source,{},'mask')


class DustJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(dust, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.source = self.root/'source.mkv'
        self.meta = {'frames':50, 'width':96, 'height':64, 'fps':'24'}
        writer = dust.Writer(self.source, self.meta)
        for _ in range(50):
            writer.write(np.full((64,96,3), 100, np.uint8))
        writer.close()
        self.values = {'detector':'legacy', 'repair':'propainter', 'license_confirmed':'true'}
        self.preflight = patch('dust_model_preflight.inspect_propainter', return_value={'ready':True, 'backend':'propainter'})
        self.preflight.start()
        self.addCleanup(self.preflight.stop)
        mask=np.zeros((64,96),np.uint8)
        mask[:,30:33]=255
        mask[9:11,70:72]=255
        with patch.object(dust,'ai_scratch_mask',return_value=mask):
            dust.run(self.source, self.values, 'mask')
        self.folder = dust.folder_for(self.source, self.values)

    def fake_model(self, url, output, video, mask, dest, width, height, fps, args, prefix):
        start = int(prefix.rsplit('_',1)[1])
        self.calls.append(start)
        count = dust.metadata(video)['frames']
        self.assertEqual(count,getattr(self,'section_lengths',{}).get(start,min(24,50-start)))
        for actual,expected in zip(dust.frames(mask),list(dust.frames(self.folder/'effective_mask.mkv'))[start:start+count],strict=True):
            np.testing.assert_array_equal(actual,expected)
        args.propainter_queued('owned-prompt')
        writer = dust.Writer(dest, dust.metadata(video))
        for image in dust.frames(video):
            writer.write(np.full_like(image, 200))
        writer.close()
        if self.cancel is not None:
            self.cancel.touch()
            self.cancel = None
        return dest

    def test_stop_retains_verified_sections_retry_skips_them_and_preserves_mask(self):
        self.calls, self.cancel = [], self.root/'stop'
        stop = self.cancel
        mask_hash = dust.digest(self.folder/'effective_mask.mkv')
        with patch('comfy_api.ensure_node_types'), patch.object(dust, 'run_propainter_chunk', side_effect=self.fake_model):
            with self.assertRaisesRegex(ValueError, 'stop requested'):
                dust.run(self.source, self.values, 'repair', cancel_file=stop)
            self.assertEqual(self.calls, [0])
            self.assertTrue((self.folder/'.model-chunks/0/complete.json').exists())
            self.assertFalse((self.folder/'repaired.mkv').exists())
            with self.assertRaises(FileNotFoundError):
                dust.manifest(self.folder,self.source,self.values,'repair')
            dust.run(self.source,self.values,'repair')
        self.assertEqual(self.calls,[0,24,48])
        self.assertEqual(dust.digest(self.folder/'effective_mask.mkv'),mask_hash)
        self.assertEqual(dust.metadata(self.folder/'repaired.mkv')['frames'],50)
        for original,result,mask in zip(dust.frames(self.folder/'input.mkv'),dust.frames(self.folder/'repaired.mkv'),dust.frames(self.folder/'effective_mask.mkv'),strict=True):
            np.testing.assert_array_equal(original[mask[:,:,0]==0],result[mask[:,:,0]==0])
            self.assertTrue(np.all(result[mask[:,:,0]>0]==200))
        dust.manifest(self.folder,self.source,self.values,'repair')

    def test_crash_between_publication_moves_resumes_without_model_rerun(self):
        self.calls,self.cancel=[],None
        rename=Path.rename
        def crash(path, dest):
            if path.name=='repair_measurements.csv' and path.parent.name=='.repair-ready':
                raise OSError('injected publication crash')
            return rename(path,dest)
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model):
            with patch.object(Path,'rename',crash),self.assertRaisesRegex(OSError,'injected'):
                dust.run(self.source,self.values,'repair')
        self.assertTrue((self.folder/'repaired.mkv').exists())
        with self.assertRaises(FileNotFoundError):
            dust.manifest(self.folder,self.source,self.values,'repair')
        before=dust.digest(self.folder/'repaired.mkv')
        with patch.object(dust,'propainter_frames',side_effect=AssertionError('must finish pending commit')):
            dust.run(self.source,self.values,'repair')
        self.assertEqual(dust.digest(self.folder/'repaired.mkv'),before)
        dust.manifest(self.folder,self.source,self.values,'repair')

    def test_changed_checkpoint_refused_without_requeue(self):
        self.calls,self.cancel=[],self.root/'stop'
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model):
            with self.assertRaises(ValueError):
                dust.run(self.source,self.values,'repair',cancel_file=self.cancel)
            (self.folder/'.model-chunks/0/model.mkv').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                dust.run(self.source,self.values,'repair')
        self.assertEqual(self.calls,[0])

    def test_unknown_submission_cannot_queue_a_duplicate(self):
        self.calls,self.cancel=[],None
        def lost(*args):
            raise RuntimeError('connection lost during submission')
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=lost) as submit:
            with self.assertRaises(RuntimeError):dust.run(self.source,self.values,'repair')
            with self.assertRaisesRegex(ValueError,'submission outcome is unknown'):
                dust.run(self.source,self.values,'repair')
        self.assertEqual(submit.call_count,1)

    def test_stop_after_input_preparation_is_retryable_without_unknown_submission(self):
        self.calls,self.cancel=[],None
        stop=self.root/'stop'
        publish=dust.publish_json
        def stop_after_inputs(path,data):
            publish(path,data)
            if path.name=='report.json' and path.parent.name.startswith('.inputs-'):
                stop.touch()
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model):
            with patch.object(dust,'publish_json',side_effect=stop_after_inputs),self.assertRaisesRegex(ValueError,'stop requested'):
                dust.run(self.source,self.values,'repair',cancel_file=stop)
            self.assertFalse((self.folder/'.model-chunks/0/job.json').exists())
            dust.run(self.source,self.values,'repair')
        self.assertEqual(self.calls,[0,24,48])

    def test_acknowledged_prompt_can_be_reattached_without_new_submission(self):
        self.calls,self.cancel=[],None
        def disconnected(*args):
            args[-2].propainter_queued('owned-prompt')
            raise RuntimeError('lost connection after acknowledgment')
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=disconnected):
            with self.assertRaises(RuntimeError):dust.run(self.source,self.values,'repair')
        def reattached(*args):
            if not self.calls:
                self.assertEqual(args[-2].propainter_prompt_id,'owned-prompt')
            return self.fake_model(*args)
        history={'owned-prompt':{'status':{'completed':True}}}
        with patch('comfy_api.ensure_node_types'),patch('comfy_api.http_json',return_value=history), \
             patch.object(dust,'run_propainter_chunk',side_effect=reattached):
            dust.run(self.source,self.values,'repair')
        self.assertEqual(self.calls,[0,24,48])

    def test_source_change_before_commit_prevents_publication_and_adoption(self):
        self.calls,self.cancel=[],None
        finish=dust.finish_repair
        def changed(*args):
            with self.source.open('ab') as stream:stream.write(b'changed source')
            return finish(*args)
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model), \
             patch.object(dust,'finish_repair',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Stale source'):
                dust.run(self.source,self.values,'repair')
        self.assertFalse((self.folder/'repaired.mkv').exists())

    def test_changed_ready_artifact_blocks_publication_without_overwrite(self):
        self.calls,self.cancel=[],None
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model), \
             patch.object(dust,'finish_repair',side_effect=OSError('crash before commit')):
            with self.assertRaises(OSError):dust.run(self.source,self.values,'repair')
        ready=self.folder/'.repair-ready/repaired.mkv'
        ready.write_bytes(b'changed artifact')
        with self.assertRaisesRegex(ValueError,'pending repair artifact'):
            dust.run(self.source,self.values,'repair')
        self.assertEqual(ready.read_bytes(),b'changed artifact')
        self.assertFalse((self.folder/'repaired.mkv').exists())

    def test_candidate_exclusion_releases_after_error(self):
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with candidate_lock(self.folder):
                with self.assertRaisesRegex(ValueError,'Another worker'):
                    dust.run(self.source,self.values,'repair')
                raise RuntimeError('injected')
        with candidate_lock(self.folder):pass

    def test_manifest_requires_completion_even_when_output_files_exist(self):
        (self.folder/'repaired.mkv').write_bytes(b'incomplete')
        with self.assertRaises(FileNotFoundError):
            dust.manifest(self.folder,self.source,self.values,'repair')

    def set_cut(self, frame):
        with (self.folder/'measurements.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        rows[frame]['abstention']='source_cut_boundary'
        (self.folder/'measurements.csv').unlink()
        dust.write_csv(self.folder/'measurements.csv',rows)
        report=json.loads((self.folder/'report.json').read_text())
        report['hashes']['measurements.csv']=dust.digest(self.folder/'measurements.csv')
        dust.publish_report(self.folder,report)

    def test_sections_keep_existing_cut_boundaries_without_padding(self):
        self.calls,self.cancel=[],None
        self.section_lengths={0:20,20:24,44:6}
        self.set_cut(20)
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model):
            dust.run(self.source,self.values,'repair')
        self.assertEqual(self.calls,[0,20,44])

    def test_single_frame_section_is_rejected_before_any_submission(self):
        self.set_cut(1)
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk') as submit:
            with self.assertRaisesRegex(ValueError,'at least two frames'):
                dust.run(self.source,self.values,'repair')
        submit.assert_not_called()

    def test_fractional_cadence_survives_section_split_and_master(self):
        self.source=self.root/'fractional.mkv'
        writer=dust.Writer(self.source,{**self.meta,'fps':'30000/1001'})
        for _ in range(50):writer.write(np.full((64,96,3),100,np.uint8))
        writer.close()
        dust.run(self.source,self.values,'mask')
        self.folder=dust.folder_for(self.source,self.values)
        self.calls,self.cancel=[],None
        with patch('comfy_api.ensure_node_types'),patch.object(dust,'run_propainter_chunk',side_effect=self.fake_model):
            dust.run(self.source,self.values,'repair')
        self.assertEqual(dust.metadata(self.folder/'repaired.mkv')['fps'],'30000/1001')

    def test_stop_only_signals_the_owned_worker_and_does_not_kill_comfy(self):
        stop=self.root/'jobs/owned.cancel'
        process=SimpleNamespace(args=['python','--cancel-file',str(stop)],poll=lambda:None)
        with patch.object(server.APP,'process',process),patch.object(server.APP,'running_stage_key','dust'), \
             patch.object(server.APP,'log',[]),patch.object(server,'terminate_process_tree') as kill:
            server.APP.stop()
        self.assertTrue(stop.is_file())
        kill.assert_not_called()

    def test_legacy_adapter_defaults_still_queue_and_resume_hook_only_reuses_owned_id(self):
        for mode in ('legacy','callback','resume'):
            saved='owned-prompt' if mode=='resume' else None
            args=SimpleNamespace(poll_seconds=2)
            if mode!='legacy':
                args.propainter_prompt_id=saved
                args.propainter_queued=Mock()
            generated=self.root/'generated.mkv';generated.write_bytes(b'fixture')
            dest=self.root/('copy-'+mode+'.mkv')
            with patch.object(cleanup_video,'propainter_prompt',return_value={'unchanged':'graph'}), \
                 patch.object(cleanup_video,'queue_prompt',return_value='new-prompt') as queue, \
                 patch.object(cleanup_video,'wait_for_prompt',return_value={}) as wait, \
                 patch.object(cleanup_video,'extract_output_files',return_value=[generated]):
                cleanup_video.run_propainter_chunk('local',self.root,self.source,self.source,dest,96,64,24,args,'prefix')
            self.assertEqual(queue.call_count,0 if saved else 1)
            wait.assert_called_once_with('local',saved or 'new-prompt',2)
            if mode!='legacy':
                args.propainter_queued.assert_called_once_with(saved or 'new-prompt')


if __name__=='__main__':unittest.main()

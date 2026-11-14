"""Real CPU round trips and fail-closed model/provenance contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np

from ai_remaster_gui import dust, server
from ai_remaster_gui.runtime_settings import base_settings
from ai_remaster_gui.commercial_policy import stage_issue


class DustTests(unittest.TestCase):
    def test_original_analysis_scope_is_explicit_and_cannot_repair(self):
        self.assertEqual(dust.options({}),dust.options({'analysis_source':'stage'}))
        with self.assertRaises(ValueError):dust.options({'analysis_source':'elsewhere'})
        with self.assertRaisesRegex(ValueError,'analysis-only'):
            dust.authorize({'analysis_source':'original'},kind='repair')
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            values={'analysis_source':'original'}
            before=dust.digest(source)
            dust.run(source,values,'mask')
            with self.assertRaises(FileNotFoundError):dust.saved_mask(source,{})
            self.assertNotEqual(dust.folder_for(source,values),dust.folder_for(source,{}))
            app=SimpleNamespace(settings={'global':{'source':str(source)}},project_path='p',
                stabilized_source_for_downstream=lambda:str(source),running_stage_key='',process=None)
            matrix=dust.matrix_state(app)
            self.assertEqual(matrix['variants'],[])
            self.assertEqual(len(matrix['original']['variants']),1)
            self.assertFalse(matrix['original']['variants'][0]['repair_complete'])
            self.assertEqual(dust.inspect(app,values,1)['frames'],3)
            with self.assertRaisesRegex(ValueError,'changed'):
                dust.status(app,values,'different-input','p')
            with self.assertRaisesRegex(ValueError,'changed'):
                dust.inspect(app,values,0,str(source),'different-project')
            self.assertEqual(dust.digest(source),before)

    def test_original_launch_bypasses_only_upstream_readiness(self):
        settings=base_settings();settings['global']['source']='original.mkv'
        values={'analysis_source':'original'}
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'section.mkv';source.write_bytes(b'original selected section')
            with patch.object(server.APP,'settings',settings),patch.object(server.APP,'quitting',False), \
                 patch.object(server.APP,'queue_running',False,create=True),patch.object(server.APP,'ensure_pipeline_source'), \
                 patch('ai_remaster_gui.media.pipeline_source_text',return_value=str(source)), \
                 patch.object(server.APP,'preprocessing_problem',side_effect=AssertionError('upstream not required')), \
                 patch.object(server.APP,'stabilized_source_for_downstream',return_value='missing-stage.mkv'), \
                 patch.object(server.APP,'_start_denoise_matrix',return_value=(True,'test launch')) as launch:
                before=repr(settings)
                self.assertTrue(server.APP.run_dust(values,'mask',str(source))[0])
                command=launch.call_args.args[1]
                self.assertEqual(command[command.index('--source')+1],str(source))
                self.assertEqual(repr(settings),before)
                with self.assertRaisesRegex(ValueError,'changed'):
                    server.APP.run_dust(values,'mask','missing-stage.mkv')

    def test_new_mask_version_reruns_detection_without_replacing_prior_repair(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            values={'detector':'legacy','repair':'telea'}
            dust.run(source,values,'mask');dust.run(source,values,'repair')
            old=dust.folder_for(source,values)
            before={p.name:dust.digest(p) for p in old.iterdir() if p.is_file()}
            fresh={**values,'mask_version':'a'*32}
            with self.assertRaises(FileNotFoundError):dust.saved_mask(source,fresh)
            with patch.object(dust,'masks_for',wraps=dust.masks_for) as detector:
                dust.run(source,fresh,'mask')
                self.assertEqual(detector.call_count,1)
                dust.run(source,fresh,'mask')  # Retry reuses this version, never a third calculation.
                self.assertEqual(detector.call_count,1)
            folder=dust.folder_for(source,fresh)
            self.assertNotEqual(old,folder)
            self.assertFalse((folder/'repaired.mkv').exists())
            self.assertEqual(before,{p.name:dust.digest(p) for p in old.iterdir() if p.is_file()})
            changed={**fresh,'repair':'propainter'}
            self.assertEqual(dust.saved_mask(source,changed)[0],folder)
            with patch.object(dust,'masks_for',side_effect=AssertionError('reuse frozen version')):
                dust.run(source,changed,'mask')
            self.assertEqual(dust.digest(folder/'effective_mask.mkv'),dust.digest(dust.folder_for(source,changed)/'effective_mask.mkv'))

    def test_mask_version_validation_preserves_legacy_identity(self):
        opts=dust.options({})
        self.assertNotIn('mask_version',opts)
        self.assertEqual(opts,dust.options({'mask_version':''}))
        for invalid in (None,False,0,[],{},'../elsewhere','A'*32,'a'*31):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):dust.options({'mask_version':invalid})
        dart=dust.options({'detector':'dart','repair':'dart','mask_version':'b'*32})
        self.assertEqual(dart['mask_version'],'b'*32)

    def test_saved_mask_playback_has_frame_locked_regions_without_repair(self):
        from ai_remaster_gui import denoise_playback
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)), \
             patch.object(denoise_playback,'_CACHE',Path(td)/'playback'),patch.object(denoise_playback,'_SECONDS',.25):
            source=Path(td)/'source.mkv';meta={'frames':8,'width':96,'height':64,'fps':'12'}
            writer=dust.Writer(source,meta)
            images=[np.full((64,96,3),30+i*20,np.uint8) for i in range(8)]
            for image in images: writer.write(image)
            writer.close()
            masks=[]
            for i,image in enumerate(images):
                mask=np.zeros((64,96),np.uint8);mask[:,8+i*8:16+i*8]=255
                masks.append((image,mask,''))
            with patch.object(dust,'masks_for',return_value=iter(masks)):
                dust.run(source,{},'mask')
            app=SimpleNamespace(stabilized_source_for_downstream=lambda:str(source))
            folder=dust.folder_for(source,{})
            before={p.name:dust.digest(p) for p in folder.iterdir() if p.is_file()}
            reel=denoise_playback.playback_segment(app,'',4,stage='dust',mask_values={})
            self.assertEqual((reel['first_frame'],reel['end_frame']),(3,6))
            capture=cv2.VideoCapture(str(denoise_playback.segment_video_path(reel['url'].split('=')[1])))
            try:
                for i in range(3,6):
                    ok,image=capture.read();self.assertTrue(ok)
                    x,y,w,h=reel['regions']['original']
                    self.assertLess(abs(float(image[y:y+h,x:x+w].mean())-(30+i*20)),5)
                    x,y,w,h=reel['regions']['mask'];mask=image[y:y+h,x:x+w]
                    self.assertGreater(float(mask[:,10+i*8:14+i*8].mean()),220)
                    self.assertLess(float(mask[:,:4].mean()),30)
            finally: capture.release()
            self.assertEqual(before,{p.name:dust.digest(p) for p in folder.iterdir() if p.is_file()})
            self.assertFalse((folder/'repaired.mkv').exists())
            with self.assertRaises(ValueError):
                denoise_playback.playback_segment(app,'',0,stage='dust',reference='original',mask_values={})
    def test_cpu_mask_launch_does_not_require_pipeline_enable_or_repair_licence(self):
        settings=base_settings();settings['global'].update(dust='false',commercial_model_guard='true')
        settings['cleanup']['ai_descratch']='true'
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.mkv';source.write_bytes(b'launch-contract')
            with patch.object(server.APP,'settings',settings),patch.object(server.APP,'quitting',False), \
                 patch.object(server.APP,'queue_running',False,create=True),patch.object(server.APP,'ensure_pipeline_source'), \
                 patch.object(server.APP,'preprocessing_problem',return_value=''),patch.object(server.APP,'stabilize_enabled',return_value=False), \
                 patch.object(server.APP,'stabilized_source_for_downstream',return_value=str(source)), \
                 patch.object(server.APP,'_start_denoise_matrix',return_value=(True,'test launch')) as launch:
                self.assertTrue(server.APP.run_dust({'detector':'temporal','repair':'propainter'},'mask')[0])
                self.assertIn('mask',launch.call_args.args[1])
                before=repr(settings)
                self.assertTrue(server.APP.run_dust({'detector':'temporal','repair':'propainter','mask_version':'1'*32},'mask')[0])
                command=launch.call_args.args[1]
                self.assertIn('"mask_version": "'+'1'*32+'"',command[command.index('--settings')+1])
                self.assertEqual(repr(settings),before)
                self.assertFalse(server.APP.run_dust({'detector':'temporal','repair':'propainter'},'repair')[0])

    def test_mask_authorization_is_independent_of_unused_repair_model(self):
        values={'detector':'temporal','repair':'propainter','license_confirmed':'false'}
        self.assertEqual(dust.authorize(values,True,'mask')['detector'],'temporal')
        with self.assertRaises(ValueError): dust.authorize(values,True,'repair')
        with self.assertRaises(ValueError): dust.authorize({'detector':'dart','repair':'dart','research_confirmed':'true'},True,'mask')

    def test_missing_dart_is_blocked_before_launch_without_installing(self):
        settings=base_settings();settings['global']['commercial_model_guard']='false'
        values={'detector':'dart','repair':'dart','research_confirmed':'true','license_confirmed':'true'}
        with patch.object(server.APP,'settings',settings),patch.object(server.APP,'quitting',False), \
             patch.object(server.APP,'queue_running',False,create=True), \
             patch('ai_remaster_gui.dart_runtime.require_installation',side_effect=ValueError('DART is not set up')) as preflight, \
             patch.object(server.APP,'_start_denoise_matrix') as launch:
            with self.assertRaisesRegex(ValueError,'not set up'):server.APP.run_dust(values,'mask')
            self.assertEqual(preflight.call_args.args[0]['dart_repo'],str(server.ROOT/'tools/dart'))
            launch.assert_not_called()

    def test_saved_dart_composition_does_not_require_reinstalling_runtime(self):
        settings=base_settings();settings['global'].update(dust='true',commercial_model_guard='false')
        values={'detector':'dart','repair':'dart','research_confirmed':'true','license_confirmed':'true'}
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.mkv';source.write_bytes(b'launch-contract')
            with patch.object(server.APP,'settings',settings),patch.object(server.APP,'quitting',False), \
                 patch.object(server.APP,'queue_running',False,create=True),patch.object(server.APP,'ensure_pipeline_source'), \
                 patch.object(server.APP,'preprocessing_problem',return_value=''),patch.object(server.APP,'stabilize_enabled',return_value=False), \
                 patch.object(server.APP,'cleanup_enabled',return_value=False), \
                 patch.object(server.APP,'stabilized_source_for_downstream',return_value=str(source)), \
                 patch.object(dust,'saved_mask') as saved, \
                 patch('ai_remaster_gui.dart_runtime.require_installation') as preflight, \
                 patch.object(server.APP,'_start_denoise_matrix',return_value=(True,'test launch')):
                self.assertTrue(server.APP.run_dust(values,'repair')[0])
                saved.assert_called_once();preflight.assert_not_called()

    def test_saved_cpu_mask_survives_repair_setting_changes_without_detection(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            values={'detector':'legacy','repair':'telea'}
            dust.run(source,values,'mask')
            app=SimpleNamespace(stabilized_source_for_downstream=lambda:str(source),settings={'dust':{}},running_stage_key='')
            changed={**values,'radius':5,'repair':'propainter'}
            with patch.object(dust,'masks_for',side_effect=AssertionError('Unexpected detector run')):
                self.assertEqual(dust.status(app,changed)['frames'],3)
                with patch.object(dust,'manifest',wraps=dust.manifest):
                    folder,report=dust.saved_mask(source,changed)
                self.assertEqual(report['hashes']['effective_mask.mkv'],dust.digest(folder/'effective_mask.mkv'))
                # A new classical repair configuration copies the same mask, without detection.
                repaired={**values,'radius':5}
                dust.run(source,repaired,'repair')
                self.assertTrue(dust.status(app,repaired)['repair_complete'])
                result=dust.folder_for(source,repaired)/'repaired.mkv'
                for missing in (False,True):
                    if missing: result.unlink()
                    else: result.write_bytes(b'changed repair')
                    status=dust.status(app,repaired)
                    self.assertEqual(status['frames'],3)
                    self.assertFalse(status['repair_complete']);self.assertTrue(status['repair_error'])
                    self.assertNotIn('repaired.mkv',status['files'])
                    rows=dust.matrix_state(app)['variants']
                    row=next(row for row in rows if row['values']['radius']==5)
                    self.assertFalse(row['repair_complete']);self.assertTrue(row['repair_error'])
                    with patch('ai_remaster_gui.frame_inspector._decode',return_value='exact.png'):
                        info=dust.inspect(app,repaired,0)
                    self.assertIn('effective_mask.mkv',info['images']);self.assertNotIn('repaired.mkv',info['images'])
                    with self.assertRaises((OSError,ValueError)):
                        dust.review_pair(app,str(result))

    def test_shared_workbench_catalog_and_provenance_bound_exact_inspection(self):
        from ai_remaster_gui import denoise_review
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            app=SimpleNamespace(stabilized_source_for_downstream=lambda:str(source),settings={'dust':{}},running_stage_key='')
            values={'detector':'legacy'}
            dust.run(source,values,'mask')
            matrix=dust.matrix_state(app)
            self.assertEqual(len(matrix['variants']),1)
            self.assertFalse(matrix['variants'][0]['output'])
            with self.assertRaises(FileNotFoundError):
                denoise_review.review_pair(app,str(dust.folder_for(source,values)/'repaired.mkv'),'dust')
            dust.run(source,values,'repair')
            row=dust.matrix_state(app)['variants'][0]
            self.assertEqual(denoise_review.review_pair(app,row['output'],'dust')[0],source)
            with patch.object(denoise_review,'_decode',return_value='exact.png'):
                info=denoise_review.inspect_denoise(app,row['output'],1,'dust')
            self.assertTrue(info['exact']);self.assertEqual(info['frame'],1)
            self.assertEqual(info['method'],'mask-gated dust')
            from ai_remaster_gui import denoise_playback
            with patch.object(denoise_playback,'_CACHE',Path(td)/'playback'):
                segment=denoise_playback.playback_segment(app,row['output'],1,stage='dust')
                self.assertEqual(segment['frames'],3)
                self.assertIn('difference',segment['regions'])
            with self.assertRaises(ValueError):
                denoise_review.review_pair(app,str(dust.folder_for(source,values)/'mask.mkv'),'dust')
            app.settings['dust']={**values,'adopted_report':row['revision']}
            self.assertTrue(dust.matrix_state(app)['variants'][0]['adopted'])
            other=Path(td)/'other.mkv';other.write_bytes(source.read_bytes())
            app.stabilized_source_for_downstream=lambda:str(other)
            self.assertEqual(dust.matrix_state(app)['variants'],[])
            with self.assertRaises(ValueError):denoise_review.review_pair(app,row['output'],'dust')

    def test_dust_actions_refuse_old_project_context_before_mutation(self):
        with patch.object(server.APP,'project_path',Path('new-project')),patch.object(server.APP,'stabilized_source_for_downstream',return_value='same.mkv'), \
             patch.object(server.APP,'process',None),patch.object(server.APP,'queue_running',False,create=True),patch.object(server.APP,'save') as save:
            with self.assertRaisesRegex(ValueError,'project/input changed'):
                server.APP.run_dust({},'mask','same.mkv','old-project')
            with self.assertRaisesRegex(ValueError,'input changed'):
                server.APP.adopt_dust({},'same.mkv','old-project')
            with patch.object(dust,'manifest'),patch.object(dust,'digest',return_value='new-report'):
                with self.assertRaisesRegex(ValueError,'candidate changed'):
                    server.APP.adopt_dust({},report_expected='old-report')
            save.assert_not_called()
    def test_defaults_order_and_model_gates(self):
        settings = base_settings()
        self.assertEqual(settings['global']['dust'], 'false')
        settings['global'].update(dust='true', stabilize='true', detail='true', commercial_model_guard='true')
        with patch.object(server.APP, 'settings', settings):
            keys = [s.key for s in server.APP.active_stages()]
            self.assertEqual(keys[keys.index('stabilize'):keys.index('detail')+1], ['stabilize', 'dust', 'detail'])
        self.assertFalse(stage_issue(settings, 'dust'))
        for method in ('propainter', 'dart'):
            settings['dust']['repair'] = method
            self.assertTrue(stage_issue(settings, 'dust'))
        for values in ({'sensitivity':'nan'}, {'radius':1.1}, {'dilate':-1}, {'repair':'dart'}, {'detector':'dart'}):
            with self.assertRaises(ValueError): dust.options(values)
        for values in ({'detector':'ridge'}, {'repair':'propainter'}, {'detector':'dart', 'repair':'dart'}):
            with self.assertRaises(ValueError): dust.authorize(values)
        with self.assertRaises(ValueError):
            dust.authorize({'repair':'propainter','license_confirmed':'true'}, True)

    def test_ridge_static_negatives_and_window_boundaries(self):
        from film_scratch_ridge import detect_sequence
        frame = np.full((96,160,3), 100, np.uint8)
        frame[:,35:38] = 255
        frame[:,80:] = 215
        self.assertTrue(all(not np.any(m) for m in detect_sequence([frame]*24)))
        with patch.object(dust, 'frames', return_value=iter([frame]*50)):
            result = list(dust.masks_for('unused', dust.options({'detector':'ridge'})))
        self.assertEqual(len(result),50)
        self.assertTrue(all(result[i][2] for i in (0,49)))
        self.assertTrue(all(not result[i][2] for i in range(1,49)))

    def test_context_stream_retains_every_input_frame_exactly_once(self):
        for count in (1,2,23,24,25,26,48,49,50,72,73):
            seq=[np.full((64,96,3),100,np.uint8) for _ in range(count)]
            for index,image in enumerate(seq): image[0,0,0]=index
            with self.subTest(count=count),patch.object(dust,'frames',return_value=iter(seq)):
                result=list(dust.masks_for('unused',dust.options({})))
                self.assertEqual([int(f[0,0,0]) for f,_,_ in result],list(range(count)))

    def test_actual_ten_bit_stage_and_native_report_integrity(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            source=Path(td)/'source.mkv'
            subprocess.run([dust.find_ffmpeg(),'-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=12:duration=1',
                '-f','lavfi','-i','sine=frequency=440:duration=1','-pix_fmt','yuv420p10le','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-c:v','ffv1','-c:a','pcm_s16le',str(source)],check=True)
            values={'detector':'legacy'}
            original_hash=dust.digest(source)
            dust.run(source,values,'mask')
            report=dust.run(source,values,'repair')
            folder=dust.folder_for(source,values)
            self.assertEqual(dust.metadata(folder/'repaired.mkv')['pixel_format'],'yuv420p10le')
            self.assertEqual(dust.metadata(folder/'repaired.mkv')['frames'],12)
            self.assertTrue(report['native_composition']['encoded_raster_verified'])
            for field in ('transfer','color_space','color_range','color_primaries'):
                self.assertEqual(dust.metadata(folder/'repaired.mkv')[field],dust.metadata(source)[field])
            self.assertEqual(dust.digest(source),original_hash)
            def audio_hash(path):
                return subprocess.check_output([dust.find_ffmpeg(),'-v','error','-i',str(path),'-map','0:a:0','-f','md5','pipe:1'])
            self.assertEqual(audio_hash(folder/'repaired.mkv'),audio_hash(source))
            dust.manifest(folder,source,values,'repair')
            (folder/'native_measurements.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError):dust.manifest(folder,source,values,'repair')

    def test_temporal_window_seam_uses_real_neighbours_without_crossing_cuts(self):
        seq=[np.full((64,96,3),100,np.uint8) for _ in range(50)]
        seq[24][20:22,20:22]=250
        with patch.object(dust,'frames',return_value=iter(seq)):
            result=list(dust.masks_for('unused',dust.options({})))
        from film_defect_masks import temporal_defect_mask
        np.testing.assert_array_equal(result[24][1],temporal_defect_mask(seq[23],seq[24],seq[25],.65))
        self.assertGreater(np.count_nonzero(result[24][1]),0)
        seq[25:]=[np.full((64,96,3),200,np.uint8) for _ in range(25)]
        with patch.object(dust,'frames',return_value=iter(seq)):
            result=list(dust.masks_for('unused',dust.options({})))
        self.assertEqual(len(result),50)
        self.assertTrue(result[24][2]);self.assertIn('source_cut_boundary',result[25][2])
        self.assertFalse(np.any(result[24][1]));self.assertFalse(np.any(result[25][1]))

    def test_native_ten_bit_actual_composition_preserves_unselected_plane_samples(self):
        from dust_native_composite import plane_shapes,plane_masks,compose
        for fmt in ('yuv420p10le','yuv422p10le','yuv444p10le','gbrp10le','gray10le'):
            with self.subTest(fmt=fmt),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                meta={'width':96,'height':64,'frames':3,'fps':'12','pixel_format':fmt}
                shapes=plane_shapes(meta)
                planes=[(np.arange(h*w,dtype=np.uint16).reshape(h,w)%1024).astype('<u2') for h,w in shapes]
                raw=b''.join(p.tobytes() for p in planes)
                subprocess.run([dust.find_ffmpeg(),'-v','error','-f','rawvideo','-pixel_format',fmt,'-s','96x64','-r','12','-i','pipe:0','-c:v','ffv1',str(root/'source.mkv')],input=raw*3,check=True)
                patch_writer=dust.Writer(root/'patch.mkv',meta)
                mask_writer=dust.Writer(root/'mask.mkv',meta)
                mask=np.zeros((64,96),np.uint8);mask[21:24,23:26]=255
                for _ in range(3):
                    patch_writer.write(np.full((64,96,3),200,np.uint8));mask_writer.write(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR))
                patch_writer.close();mask_writer.close()
                counts=compose(dust.find_ffmpeg(),root/'source.mkv',root/'patch.mkv',root/'mask.mkv',root/'output.mkv',meta)
                self.assertEqual(dust.metadata(root/'output.mkv')['pixel_format'],fmt)
                output=subprocess.check_output([dust.find_ffmpeg(),'-v','error','-i',str(root/'output.mkv'),'-pix_fmt',fmt,'-f','rawvideo','pipe:1'])
                self.assertEqual(len(output),len(raw)*3)
                offset=0
                for original,support in zip(planes,plane_masks(mask>0,meta),strict=True):
                    size=original.size*2
                    actual=np.frombuffer(output[offset:offset+size],'<u2').reshape(original.shape)
                    np.testing.assert_array_equal(actual[~support],original[~support])
                    self.assertTrue(np.any(actual[support]!=original[support]))
                    offset+=size
                self.assertTrue(all(v==0 for row in counts for k,v in row.items() if k.endswith('outside_samples')))

    def test_actual_lossless_masks_repair_and_tamper(self):
        with tempfile.TemporaryDirectory() as td, patch.object(dust, 'ROOT', Path(td)):
            source=Path(td)/'input.mkv'
            subprocess.run([dust.find_ffmpeg(), '-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=12:duration=1',
                '-c:v','ffv1',str(source)],check=True)
            values={'detector':'temporal'}
            before=dust.digest(source)
            report=dust.run(source,values,'mask')
            folder=dust.folder_for(source,values)
            self.assertEqual(report['metadata']['frames'],12)
            self.assertFalse(report['repair_complete'])
            report=dust.run(source,values,'repair')
            self.assertTrue(report['repair_complete'])
            for original, output, mask in zip(dust.frames(folder/'input.mkv'),dust.frames(folder/'repaired.mkv'),dust.frames(folder/'effective_mask.mkv'),strict=True):
                np.testing.assert_array_equal(original[mask[:,:,0]==0],output[mask[:,:,0]==0])
            self.assertEqual(before,dust.digest(source))
            dust.run(source,values,'repair')  # exact reuse
            (folder/'mask.mkv').write_bytes(b'changed')
            with self.assertRaises(ValueError): dust.manifest(folder,source,values,'repair')

    def test_missing_input_is_pending_not_replaced_by_original(self):
        with tempfile.TemporaryDirectory() as td, patch.object(dust, 'ROOT', Path(td)):
            path=dust.output_for(str(Path(td)/'missing.mkv'),{})
            self.assertIn('repaired.mkv',path)

    def test_frozen_mask_shared_with_propainter_and_explicit_adoption(self):
        with tempfile.TemporaryDirectory() as td, patch.object(dust, 'ROOT', Path(td)):
            source=Path(td)/'input.mkv'
            subprocess.run([dust.find_ffmpeg(),'-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=12:duration=1','-c:v','ffv1',str(source)],check=True)
            telea={'detector':'legacy'}
            dust.run(source,telea,'mask')
            neural={'detector':'legacy','repair':'propainter','license_confirmed':'true'}
            with patch.object(dust,'ai_scratch_mask',side_effect=AssertionError('must reuse frozen mask')):
                dust.run(source,neural,'mask')
            folder=dust.folder_for(source,neural)
            self.assertEqual(dust.digest(folder/'mask.mkv'),dust.digest(dust.folder_for(source,telea)/'mask.mkv'))
            def fake(*args):
                for image in dust.frames(folder/'input.mkv'):
                    yield np.full_like(image,127)
            with patch.object(dust,'propainter_frames',side_effect=fake),patch('dust_model_preflight.inspect_propainter',return_value={'ready':True,'backend':'propainter','test_double':True}):
                dust.run(source,neural,'repair')
            settings=base_settings()
            settings['global'].update(source=str(source),dust='true')
            settings['dust'].update({k:str(v) for k,v in neural.items()})
            with patch.object(server.APP,'settings',settings),patch.object(server.APP,'stabilized_source_for_downstream',return_value=str(source)), \
                 patch.object(server.APP,'save'),patch.object(server.APP,'hydrate_stage_inputs'), \
                 patch.object(server.APP,'process',None),patch.object(server.APP,'queue_running',False,create=True):
                with self.assertRaises(ValueError): server.APP.validate_dust_adoption()
                server.APP.adopt_dust(neural)
                server.APP.validate_dust_adoption()
                settings['dust']['radius']='4'
                with self.assertRaises((OSError,ValueError)): server.APP.validate_dust_adoption()


if __name__=='__main__': unittest.main()

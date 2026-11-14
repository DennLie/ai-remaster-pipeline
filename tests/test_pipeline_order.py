import json
import unittest
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import pipeline_order,server
from ai_remaster_gui.runtime_settings import base_settings


class PipelineOrderTests(unittest.TestCase):
    def settings(self,order=None):
        values=base_settings()
        values['global'].update(source='input/pipeline-test.mkv',colorize='true',expand_outpaint='true',upscale='true',add_soundtrack='true',
            pipeline_order=','.join(order or pipeline_order.STAGES))
        values['upscale']['auto_target']='false'
        return values

    def build(self,settings):
        return pipeline_order.build(settings,'raw',pipeline_order.order(settings,()),lambda key,source,bindings:source+'/'+key)

    def test_complete_order_validation_and_read_only_migration(self):
        settings={'global':{'video_order':'kept'}}
        video=list(reversed(pipeline_order.VIDEO_STAGES))
        self.assertEqual(pipeline_order.order(settings,video),video+list(pipeline_order.STAGES[7:]))
        self.assertNotIn('pipeline_order',settings['global'])
        for value in ('',None,[],','.join(pipeline_order.STAGES[:-1]),'dust,'*14):
            with self.assertRaises(ValueError):pipeline_order.order({'global':{'pipeline_order':value}},(),strict=True)

    def test_default_bindings_and_metadata_phases_do_not_replace_video(self):
        plan=self.build(self.settings());rows=plan['nodes']
        self.assertFalse(plan['problems'])
        self.assertEqual(rows['shots']['input'],rows['references']['input'])
        self.assertEqual(rows['colour']['bindings']['manifest'],rows['shots']['bindings']['manifest'])
        self.assertEqual(rows['recomp']['bindings']['outpainted_video'],rows['outpaint']['output'])
        self.assertEqual(rows['recomp']['bindings']['source'],rows['outpaint']['input'])

    def test_upscale_before_colour_and_audio_after_upscale_are_real_bindings(self):
        ordered=[*pipeline_order.VIDEO_STAGES,'upscale','outpaint','shots','references','colour','recomp','audio']
        plan=self.build(self.settings(ordered));rows=plan['nodes']
        self.assertFalse(plan['problems'])
        self.assertEqual(rows['outpaint']['input'],rows['upscale']['output'])
        self.assertEqual(rows['colour']['bindings']['source'],rows['outpaint']['output'])
        self.assertEqual(rows['audio']['input'],rows['recomp']['output'])
        self.assertEqual(plan['output'],rows['audio']['output'])

    def test_stale_reference_and_canvas_pairings_are_not_silently_reused(self):
        ordered=list(pipeline_order.STAGES);ordered.remove('upscale');ordered.insert(ordered.index('colour'),'upscale')
        plan=self.build(self.settings(ordered))
        self.assertIn('exact video',plan['nodes']['colour']['problem'])
        self.assertTrue(plan['problems'])

    def test_all_reference_permutations_reject_consumers_before_producers(self):
        import itertools
        for triple in itertools.permutations(('shots','references','colour')):
            ordered=[*pipeline_order.VIDEO_STAGES,'outpaint',*triple,'recomp','audio','upscale']
            self.assertEqual(bool(self.build(self.settings(ordered))['problems']),triple!=('shots','references','colour'))

    def test_colour_can_be_recomposed_before_later_outpainting(self):
        ordered=[*pipeline_order.VIDEO_STAGES,'shots','references','colour','recomp','outpaint','audio','upscale']
        plan=self.build(self.settings(ordered))
        self.assertFalse(plan['problems'])
        self.assertEqual(plan['nodes']['recomp']['bindings']['outpainted_video'],'')

    def test_server_commands_get_planned_sources_and_ignore_stale_derived_settings(self):
        ordered=[*pipeline_order.VIDEO_STAGES,'upscale','outpaint','shots','references','colour','recomp','audio']
        settings=self.settings(ordered)
        settings['global'].update({key:'true' for key in pipeline_order.VIDEO_STAGES})
        settings['references']['manifest']='old.json';settings['shots']['outpainted_video']='old.mkv'
        settings['recomp'].update(source='old.mkv',colorized_video='old-colour.mkv')
        with ExitStack() as stack:
            stack.enter_context(patch.object(server.APP,'settings',settings))
            stack.enter_context(patch.object(server.APP,'apply_auto_upscale_target'))
            stack.enter_context(patch.object(server,'ordered_video_output',side_effect=lambda settings,key,source:source+'/'+key))
            stack.enter_context(patch.object(server,'outpaint_output_for',side_effect=lambda source,*args,**kwargs:source+'/outpaint.mkv'))
            stack.enter_context(patch.object(server,'manifest_for_outpainted',side_effect=lambda source:source+'/shots.json'))
            stack.enter_context(patch.object(server,'colorized_output_for_manifest',side_effect=lambda manifest,*args:manifest+'/colour.mkv'))
            stack.enter_context(patch.object(server,'recomposition_output_for',side_effect=lambda source,*args:source+'/recomp.mp4'))
            stack.enter_context(patch.object(server,'upscale_output_for',side_effect=lambda source,*args:source+'/upscale.mkv'))
            stack.enter_context(patch.object(server,'soundtrack_output_for',side_effect=lambda source,*args:source+'/audio.mp4'))
            plan=server.APP.pipeline_plan();self.assertFalse(plan['problems'])
            server.APP.hydrate_pipeline_bindings()
            rows=plan['nodes']
            for key in pipeline_order.VIDEO_STAGES:
                command=server.APP.local_command_for(key)
                self.assertEqual(command[command.index('--source')+1],rows[key]['input'])
            self.assertEqual(server.APP.soundtrack_source_for(),rows['audio']['input'])
            self.assertEqual(server.APP.upscale_input_for(),rows['upscale']['input'])
            self.assertEqual(server.outpaint_source_for_settings(settings),rows['outpaint']['input'])
            for key,flag in (('shots','--source-video'),('references','--manifest'),('colour','--manifest'),('audio','--input'),('upscale','--input'),('recomp','--source')):
                cmd=server.APP.local_command_for(key)
                expected=rows[key]['bindings']['manifest'] if flag=='--manifest' else rows[key]['bindings']['source'] if key=='recomp' else rows[key]['input']
                self.assertEqual(cmd[cmd.index(flag)+1],expected)
                if key=='shots':self.assertEqual(cmd[cmd.index('--output-manifest')+1],rows[key]['bindings']['manifest'])
            self.assertNotIn('old.json',settings['references']['manifest'])
            self.assertEqual([s.key for s in server.APP.active_stages()],list(plan['nodes']))

    def test_upscale_uses_only_earlier_shots_and_view_avoids_legacy_fallbacks(self):
        from ai_remaster_gui import references
        for shots_first in (False,True):
            ordered=list(pipeline_order.STAGES)
            if not shots_first:
                ordered.remove('upscale');ordered.insert(0,'upscale')
            settings=self.settings(ordered)
            settings['references']['manifest']='unrelated-old.csv'
            with patch.object(server.APP,'settings',settings),patch.object(server.APP,'apply_auto_upscale_target'):
                plan=server.APP.hydrate_pipeline_bindings()
                manifest=plan['nodes']['shots']['bindings']['manifest'] if shots_first else ''
                self.assertEqual(server.APP.upscale_shot_manifest(),manifest)
                command=server.APP.local_command_for('upscale')
                if shots_first:self.assertEqual(command[command.index('--shot-manifest')+1],manifest)
                else:self.assertNotIn('--shot-manifest',command)
                with patch.object(references,'shot_rows',return_value=[]):
                    view=references.shot_views(settings,'upscale')
                self.assertEqual(view['upscale_manifest'],manifest)
                self.assertEqual(view['shots_manifest'],plan['nodes']['shots']['bindings']['manifest'])

    def test_shot_producer_stamp_survives_publication_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.mkv';source.write_bytes(b'fixture')
            settings=self.settings()
            settings['global'].update(source=str(source),expand_outpaint='false',upscale='false',add_soundtrack='false')
            settings['global'].update({key:'false' for key in pipeline_order.VIDEO_STAGES})
            with patch.object(server.APP,'settings',settings),patch.object(server.stage_stamps,'STAMP_DIR',Path(td)/'stamps'), \
                 patch.object(server,'video_metrics',return_value={}),patch.object(server,'manifest_for_outpainted',return_value=str(Path(td)/'shots.csv')):
                manifest=server.resolve(server.APP.expected_outputs('shots')[0])
                launch=server.APP.stage_stamp('shots')
                manifest.write_text(f'# source_video={source}\nend\n1\n')
                server.APP.record_stage_stamp('shots',launch)
                self.assertTrue(server.APP.stage_is_current('shots'))
                source.write_bytes(b'changed source fixture')
                self.assertFalse(server.APP.stage_is_current('shots'))

    def test_invalid_dependencies_block_before_any_worker_or_network_action(self):
        ordered=list(pipeline_order.STAGES);ordered.remove('references');ordered.insert(0,'references')
        with patch.object(server.APP,'settings',self.settings(ordered)),patch.object(server,'video_metrics',return_value={}),patch.object(server.subprocess,'Popen') as popen:
            self.assertFalse(server.APP.run_all()[0]);self.assertFalse(server.APP.run_stage('colour')[0]);popen.assert_not_called()

    def test_stale_manifest_blocks_direct_reference_and_guide_model_entry_points(self):
        settings=self.settings()
        with patch.object(server.APP,'settings',settings),patch.object(server.APP,'pipeline_plan',return_value={'problems':[], 'nodes':{'references':{'bindings':{'manifest':'current.csv'}}}}), \
             patch.object(server,'ensure_comfy_available_for_stage') as comfy,patch.object(server.subprocess,'Popen') as worker:
            self.assertFalse(server.APP.run_reference_regeneration('old.csv',0)[0])
            self.assertFalse(server.APP.run_reference_edit_preview('old.csv',0,'test')[0])
            self.assertFalse(server.APP.run_outpaint_chunk(0)[0])
            self.assertFalse(server.APP.run_guide_edit_preview(0,0,'test')[0])
            comfy.assert_not_called();worker.assert_not_called()

    def test_actual_manifest_source_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            manifest=Path(td)/'shots.csv';manifest.write_text('fixture')
            plan={'problems':[],'nodes':{'references':{'input':'new.mkv','output':'new.mkv','bindings':{'source':'new.mkv','manifest':str(manifest)},'problem':''}}}
            with patch.object(server.APP,'settings',self.settings()),patch.object(server.APP,'pipeline_plan',return_value=plan),patch.object(server,'manifest_source_video',return_value='old.mkv'):
                self.assertIn('another video',server.APP.preprocessing_problem('references'))

    def test_recomposition_accepts_resolution_change_but_rejects_branch_retiming(self):
        from ai_remaster_gui import frame_inspector
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.mkv';canvas=Path(td)/'canvas.mkv'
            source.write_bytes(b'synthetic metadata fixture');canvas.write_bytes(b'synthetic metadata fixture')
            row={'input':str(canvas),'output':'final.mp4','bindings':{'source':str(source),'outpainted_video':str(canvas),'colorized_video':''},'problem':''}
            with patch.object(server.APP,'settings',self.settings()),patch.object(server.APP,'pipeline_plan',return_value={'nodes':{'recomp':row},'problems':[]}):
                before={'frames':50,'fps_fraction':'24','width':256,'height':192}
                after={**before,'width':512,'height':288}
                with patch.object(frame_inspector,'_metadata',side_effect=[before,after]):
                    self.assertEqual(server.APP.preprocessing_problem('recomp'),'')
                with patch.object(frame_inspector,'_metadata',side_effect=[before,{**after,'frames':49}]):
                    self.assertIn('cadence do not match',server.APP.preprocessing_problem('recomp'))

    def test_scoped_manifest_separates_same_basename_sources_without_mutating_settings(self):
        settings=self.settings();settings['global'].update(expand_outpaint='false',upscale='false',add_soundtrack='false')
        settings['global'].update({key:'false' for key in pipeline_order.VIDEO_STAGES})
        with patch.object(server.APP,'settings',settings):
            settings['global']['source']='input/one/film.mkv';before=json.dumps(settings)
            one=server.APP.pipeline_plan()['nodes']['shots']['bindings']['manifest']
            self.assertEqual(json.dumps(settings),before)
            settings['global']['source']='input/two/film.mkv'
            two=server.APP.pipeline_plan()['nodes']['shots']['bindings']['manifest']
            self.assertNotEqual(one,two)

    def test_keep_shots_does_not_override_provenance_after_reordering(self):
        plan={'problems':[],'nodes':{'shots':{}}}
        with patch.object(server.APP,'settings',self.settings()),patch.object(server.APP,'pipeline_plan',return_value=plan), \
             patch.object(server.APP,'existing_outputs',return_value=['old.csv']),patch.object(server.APP,'stage_is_current',return_value=False):
            ok,message=server.APP.run_all(keep_shots=True)
            self.assertFalse(ok);self.assertIn('Cannot keep stale',message)

    def test_settings_save_is_project_bound_and_preserves_previous_order_fields(self):
        settings=self.settings();settings['global'].pop('pipeline_order');old=json.dumps(settings)
        with patch.object(server.APP,'settings',settings),patch.object(server.APP,'process',None),patch.object(server.APP,'queue_running',False,create=True), \
             patch.object(server.APP,'project_path',Path('project-A')),patch.object(server.APP,'save'),patch.object(server.APP,'hydrate_stage_inputs'):
            with self.assertRaisesRegex(ValueError,'Project changed'):
                server.APP.update_settings('global',{'pipeline_order':','.join(pipeline_order.STAGES)},project_expected='project-B')
            self.assertEqual(json.dumps(settings),old)
            server.APP.update_settings('global',{'pipeline_order':','.join(pipeline_order.STAGES)},project_expected='project-A')
            self.assertIn('preprocessing_order',settings['global'])


if __name__=='__main__':unittest.main()

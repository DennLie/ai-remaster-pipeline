import itertools
import unittest
import tempfile
import json
from types import SimpleNamespace
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

from ai_remaster_gui import server,video_order
from ai_remaster_gui.runtime_settings import base_settings


class VideoOrderTests(unittest.TestCase):
    def test_cleanup_resolution_change_is_allowed_but_retiming_is_not(self):
        from ai_remaster_gui import denoise_review
        import artifact_ids as aid
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.mkv';source.write_bytes(b'input fixture')
            identity={'test_fixture':True}
            output=Path(td)/aid.artifact_name(aid.source_word(source.name),'cleanup',identity,'mkv')
            output.write_bytes(b'output fixture')
            stat=source.stat()
            provenance={'tool':'cleanup_video.py','source':str(source),'source_fingerprint':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},'identity':identity,'intermediate_profile':'lossless'}
            output.with_suffix('.mkv.sig.json').write_text(json.dumps(provenance))
            app=SimpleNamespace(settings={'cleanup':{},'cloud':{'intermediate_format':'lossless'}},processing_input=lambda stage:str(source),cleanup_output=lambda:str(output))
            before={'frames':3,'fps_fraction':'12','width':1920,'height':1080}
            after={**before,'width':1280,'height':720}
            with patch.object(denoise_review,'_metadata',side_effect=[before,after]):
                self.assertEqual(denoise_review.selected_output(app,'cleanup'),output)
            with patch.object(denoise_review,'_metadata',side_effect=[before,{**after,'frames':2}]):
                with self.assertRaisesRegex(ValueError,'different timeline'):
                    denoise_review.selected_output(app,'cleanup')

    def test_pending_chain_never_surfaces_old_downstream_outputs(self):
        settings=base_settings();settings['global']['video_order']=','.join(video_order.STAGES)
        with patch.object(server.APP,'settings',settings),patch.object(server.APP,'preprocessing_problem',return_value='Run Dust first'), \
             patch.object(server.APP,'preprocessing_upstream',return_value=['dust','detail']),patch.object(server.APP,'processing_input',return_value='missing-final.mkv'), \
             patch.object(server.APP,'upscale_input_for',side_effect=AssertionError('do not select an unrelated cached output')):
            result=server.APP.output_selection_state()
            self.assertEqual(result['path'],'missing-final.mkv')
            self.assertEqual(result['kind'],'detail_pending')

    def test_real_cpu_chain_reorder_invalidates_adoption_without_deleting_outputs(self):
        import numpy as np
        from ai_remaster_gui import dust
        import detail_enhance_video as detail
        settings=base_settings()
        settings['global'].update({key:'false' for key in video_order.STAGES})
        settings['global'].update(detail='true',dust='true',video_order='detail,dust,denoise,deflicker,enhancement,cleanup,stabilize')
        settings['detail'].update(method='unsharp',preset='gentle',mix='100')
        settings['cloud']['intermediate_format']='lossless'
        with tempfile.TemporaryDirectory() as td,patch.object(server,'ROOT',Path(td)),patch.object(dust,'ROOT',Path(td)), \
             patch.object(server.APP,'settings',settings),patch.object(server.APP,'process',None),patch.object(server.APP,'queue_running',False,create=True), \
             patch.object(server.APP,'save'),patch.object(server.APP,'hydrate_stage_inputs'):
            source=Path(td)/'source.mkv';settings['global']['source']=str(source)
            writer=dust.Writer(source,{'frames':3,'width':96,'height':64,'fps':'12'})
            for _ in range(3):writer.write(np.full((64,96,3),100,np.uint8))
            writer.close()
            target=server.resolve(server.APP.detail_output())
            detail.run(source,target,method='unsharp',preset='gentle',mix=100,intermediate_profile='lossless',ffmpeg=dust.find_ffmpeg(),force=False)
            self.assertEqual(server.APP.preprocessing_problem('dust'),'')
            values={'detector':'legacy'}
            dust.run(target,values,'mask');dust.run(target,values,'repair')
            server.APP.adopt_dust(values)
            server.APP.validate_dust_adoption()
            old=server.resolve(server.APP.dust_output());sha=dust.digest(old)
            self.assertEqual(server.outpaint_source_for_settings(settings),server.APP.dust_output())
            settings['global']['video_order']='dust,detail,denoise,deflicker,enhancement,cleanup,stabilize'
            with self.assertRaises((OSError,ValueError)):server.APP.validate_dust_adoption()
            self.assertEqual(dust.digest(old),sha)
            self.assertIn('Dust',server.APP.preprocessing_problem('detail'))

    def test_all_permutations_and_enabled_subsets_fold_in_order(self):
        keys=video_order.STAGES
        for ordered in itertools.permutations(keys):
            for bits in itertools.product((False,True),repeat=len(keys)):
                settings={'global':{k:'true' if enabled else 'false' for k,enabled in zip(keys,bits)}}
                seen=[]
                result=video_order.input_for(settings,None,('source',),ordered,lambda key,source:seen.append(key) or (*source,key))
                expected=tuple(k for k in ordered if settings['global'][k]=='true')
                self.assertEqual(result,('source',*expected))
                self.assertEqual(tuple(seen),expected)

    def test_migration_is_read_only_and_invalid_orders_never_silently_fall_back(self):
        settings={'global':{'preprocessing_order':'enhancement,denoise,deflicker'}}
        self.assertEqual(video_order.order(settings,['enhancement','denoise','deflicker']),['enhancement','denoise','deflicker','cleanup','stabilize','dust','detail'])
        self.assertNotIn('video_order',settings['global'])
        for stored in ('',','.join(video_order.STAGES[:-1]),'dust,'*7,None,[]):
            with self.assertRaises(ValueError):video_order.order({'global':{'video_order':stored}},(),strict=True)

    def test_commands_inputs_outputs_and_terminal_boundary_use_one_chain(self):
        settings=base_settings();settings['global'].update(source='input/order.mkv',**{k:'true' for k in video_order.STAGES})
        with ExitStack() as stack:
            stack.enter_context(patch.object(server.APP,'settings',settings))
            stack.enter_context(patch.object(server,'comparison_output_ready',return_value=True))
            mappings=[(server,'denoise_output_for','denoise'),(server.deflicker_stage,'output_for','deflicker'),
                (server.enhancement_stage,'output_for','enhancement'),(server,'cleanup_output_for','cleanup'),
                (server,'stabilize_output_for','stabilize'),(server.dust_stage,'output_for','dust'),(server,'detail_output_for','detail')]
            for module,name,key in mappings:
                stack.enter_context(patch.object(module,name,side_effect=lambda source,*args,k=key:source+'/'+k))
            orders=[video_order.STAGES,tuple(reversed(video_order.STAGES))]
            orders.extend(video_order.STAGES[i:]+video_order.STAGES[:i] for i in range(1,7))
            for order in orders:
                settings['global']['video_order']=','.join(order)
                expected='input/order.mkv'
                for key in order:
                    self.assertEqual(server.APP.processing_input(key),expected)
                    command=server.APP.command_for(key)
                    self.assertEqual(command[command.index('--source')+1],expected)
                    expected+='/'+key
                self.assertEqual(server.outpaint_source_for_settings(settings),expected)
                self.assertEqual(server.APP.detailed_source_for_downstream(),expected)
                self.assertEqual([s.key for s in server.APP.active_stages() if s.key in order],list(order))

    def test_strict_upstream_checks_only_predecessors_and_never_use_original(self):
        settings=base_settings();settings['global'].update(source='input/order.mkv',dust='true',stabilize='true',video_order='dust,stabilize,cleanup,detail,denoise,enhancement,deflicker')
        with patch.object(server.APP,'settings',settings),patch.object(server.APP,'queue_running',False,create=True),patch.object(server,'comparison_output_ready',return_value=False):
            self.assertEqual(server.APP.preprocessing_upstream('dust'),[])
            self.assertEqual(server.APP.preprocessing_upstream('stabilize'),['dust'])
            self.assertIn('Dust',server.APP.preprocessing_problem('stabilize'))
            self.assertFalse(server.APP.preprocessing_uses_original('stabilize'))
            self.assertNotEqual(server.APP.processing_input('stabilize'),'input/order.mkv')

    def test_save_full_order_is_atomic_with_respect_to_running_jobs(self):
        settings=base_settings();order=','.join(reversed(video_order.STAGES))
        with patch.object(server.APP,'settings',settings),patch.object(server.APP,'process',None),patch.object(server.APP,'queue_running',False,create=True),patch.object(server.APP,'save') as save,patch.object(server.APP,'hydrate_stage_inputs'):
            server.APP.update_settings('global',{'video_order':order})
            self.assertEqual(settings['global']['video_order'],order)
            save.assert_called_once()
            server.APP.queue_running=True
            with self.assertRaisesRegex(ValueError,'Stop processing'):
                server.APP.update_settings('global',{'video_order':','.join(video_order.STAGES)})
            self.assertEqual(settings['global']['video_order'],order)
            server.APP.queue_running=False
            with patch.object(server.APP,'project_path',Path('new-project')):
                with self.assertRaisesRegex(ValueError,'Project changed'):
                    server.APP.update_settings('global',{'video_order':','.join(video_order.STAGES)},project_expected='old-project')
            self.assertEqual(settings['global']['video_order'],order)


if __name__=='__main__':unittest.main()

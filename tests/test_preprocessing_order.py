import itertools
import unittest
from unittest.mock import patch, Mock

from ai_remaster_gui import server
from ai_remaster_gui.runtime_settings import base_settings
from ai_remaster_gui.denoise_review import comparison_source


class PreprocessingOrderTests(unittest.TestCase):
    def test_phase_progress_reuses_one_snapshot_without_rechecking_outputs(self):
        from ai_remaster_gui.models import STAGES
        stages = [item for item in STAGES if item.key in ('denoise', 'deflicker', 'enhancement')]
        app = server.APP
        rows = [{'stage': item.title, 'latest': 'ready.mkv' if index == 0 else ''} for index, item in enumerate(stages)]
        with patch.object(app, 'active_stages', return_value=stages), patch.object(app, 'estimate_running_progress', return_value={}), \
             patch.object(app, 'running_stage_key', ''), patch.object(app, 'progress', return_value=rows) as progress:
            direct = app.phase_progress()
            progress.assert_called_once_with()
            progress.reset_mock()
            self.assertEqual(app.phase_progress(rows), direct)
            progress.assert_not_called()
            self.assertEqual(app.phase_progress([])['global']['percent'], 0)

    def test_progress_does_not_call_an_existing_but_invalid_result_ready(self):
        from ai_remaster_gui.models import STAGES
        stage=next(item for item in STAGES if item.key=='denoise')
        app=server.APP
        with patch.object(app,'active_stages',return_value=(stage,)), \
             patch.object(app,'expected_outputs',return_value=['stale.mkv']), \
             patch.object(server,'resolve') as resolve, \
             patch.object(server,'comparison_output_ready',return_value=False):
            resolve.return_value.exists.return_value=True
            self.assertEqual(app.progress()[0]['status'],'Waiting')
            self.assertEqual(app.phase_progress()['stages'][0]['percent'],0)

    def test_all_orders_and_enabled_subsets_share_the_same_input_chain(self):
        settings = base_settings()
        settings['global']['source'] = 'input/order-test.mkv'
        settings['enhancement']['method'] = 'cas'
        functions = {'denoise': server.denoise_output_for, 'deflicker': server.deflicker_stage.output_for,
                     'enhancement': server.enhancement_stage.output_for}
        with patch.object(server.APP, 'settings', settings), patch.object(server.APP, 'queue_running', True, create=True):
            for order in itertools.permutations(server.PREPROCESSING_STAGES):
                settings['global']['preprocessing_order'] = ','.join(order)
                for enabled in itertools.product(('false', 'true'), repeat=3):
                    settings['global'].update(zip(order, enabled))
                    expected = server.pipeline_source_text(settings)
                    for key in order:
                        self.assertEqual(server.APP.processing_input(key), expected)
                        self.assertEqual(comparison_source(server.APP, key), expected)
                        command = server.APP.command_for(key)
                        self.assertEqual(command[command.index('--source') + 1], expected)
                        if settings['global'][key] == 'true':
                            expected = functions[key](expected, settings[key], settings['cloud']['intermediate_format'])
                    self.assertEqual(server.APP.enhanced_source_for_downstream(), expected)
                    settings['global'].update(cleanup='false', stabilize='false', detail='false')
                    self.assertEqual(server.outpaint_source_for_settings(settings), expected)
                    active = [stage.key for stage in server.APP.active_stages() if stage.key in order]
                    self.assertEqual(active, [key for key in order if settings['global'][key] == 'true'])

    def test_order_validation_and_old_project_defaults(self):
        self.assertEqual(server.preprocessing_order({}), list(server.PREPROCESSING_STAGES))
        for value in ('denoise,denoise,enhancement', 'cleanup,deflicker,denoise', '', [], None):
            with self.assertRaises(ValueError):
                server.preprocessing_order({'global': {'preprocessing_order': value}}, strict=True)

    def test_order_change_is_saved_without_deleting_results_and_blocked_while_busy(self):
        app = server.APP
        settings = base_settings()
        with patch.object(app, 'settings', settings), patch.object(app, 'process', None), \
             patch.object(app, 'queue_running', False, create=True), patch.object(app, 'save') as save, \
             patch.object(app, 'hydrate_stage_inputs') as hydrate:
            app.update_settings('global', {'preprocessing_order': 'deflicker,denoise,enhancement'})
            self.assertEqual(server.preprocessing_order(settings)[0], 'deflicker')
            hydrate.assert_called_once_with('global')
            save.assert_called_once()
            app.queue_running = True
            with self.assertRaisesRegex(ValueError, 'Stop processing'):
                app.update_settings('global', {'preprocessing_order': 'denoise,deflicker,enhancement'})
            app.queue_running = False
            app.process = Mock()
            app.process.poll.return_value = None
            with self.assertRaisesRegex(ValueError, 'Stop processing'):
                app.update_settings('global', {'preprocessing_order': 'denoise,deflicker,enhancement'})
            self.assertEqual(server.preprocessing_order(settings)[0], 'deflicker')

    def test_upstream_checks_follow_order_not_stage_names(self):
        settings = base_settings()
        settings['global'].update(denoise='true', deflicker='true', enhancement='true',
                                  preprocessing_order='enhancement,deflicker,denoise')
        with patch.object(server.APP, 'settings', settings), patch.object(server.APP, 'queue_running', True, create=True), patch.object(server, 'comparison_output_ready', return_value=False):
            self.assertEqual(server.APP.preprocessing_problem('enhancement'), '')
            self.assertIn('Detail Enhancement / Deblur', server.APP.preprocessing_problem('deflicker'))
            self.assertEqual(server.APP.preprocessing_upstream('denoise'), ['enhancement', 'deflicker'])
            context = server.APP.processing_input_state('denoise')
            self.assertEqual(context['from_stage'], 'deflicker')
            self.assertEqual(context['label'], 'Light Deflicker')

    def test_standalone_stage_uses_original_but_whole_queue_stays_strict(self):
        settings=base_settings()
        settings['global'].update(source='input/original.mkv',denoise='true',deflicker='true',enhancement='true')
        with patch.object(server.APP,'settings',settings), patch.object(server.APP,'queue_running',False,create=True), \
             patch.object(server,'comparison_output_ready',return_value=False):
            self.assertEqual(server.APP.processing_input('enhancement'),'input/original.mkv')
            self.assertEqual(server.APP.preprocessing_upstream('enhancement'),[])
            self.assertEqual(server.APP.preprocessing_problem('enhancement'),'')
            self.assertTrue(server.APP.processing_input_state('enhancement')['original_fallback'])
            command=server.APP.command_for('enhancement')
            self.assertEqual(command[command.index('--source')+1],'input/original.mkv')
            server.APP.queue_running=True
            self.assertTrue(server.APP.preprocessing_problem('enhancement'))
            self.assertNotEqual(server.APP.processing_input('enhancement'),'input/original.mkv')


if __name__ == '__main__':
    unittest.main()

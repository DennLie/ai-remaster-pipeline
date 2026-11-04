import itertools
import unittest
from unittest.mock import patch, Mock

from ai_remaster_gui import server
from ai_remaster_gui.runtime_settings import base_settings
from ai_remaster_gui.denoise_review import comparison_source


class PreprocessingOrderTests(unittest.TestCase):
    def test_all_orders_and_enabled_subsets_share_the_same_input_chain(self):
        settings = base_settings()
        settings['global']['source'] = 'input/order-test.mkv'
        settings['enhancement']['method'] = 'cas'
        functions = {'denoise': server.denoise_output_for, 'deflicker': server.deflicker_stage.output_for,
                     'enhancement': server.enhancement_stage.output_for}
        with patch.object(server.APP, 'settings', settings):
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
        with patch.object(server.APP, 'settings', settings), patch.object(server, 'comparison_output_ready', return_value=False):
            self.assertEqual(server.APP.preprocessing_problem('enhancement'), '')
            self.assertIn('Detail Enhancement / Deblur', server.APP.preprocessing_problem('deflicker'))
            self.assertEqual(server.APP.preprocessing_upstream('denoise'), ['enhancement', 'deflicker'])
            context = server.APP.processing_input_state('denoise')
            self.assertEqual(context['from_stage'], 'deflicker')
            self.assertEqual(context['label'], 'Light Deflicker')


if __name__ == '__main__':
    unittest.main()

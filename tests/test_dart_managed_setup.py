"""Installer contracts with temporary files and mocked commands; no downloads/GPU."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import setup_dart as setup
from ai_remaster_gui import dart_runtime,dust


class ManagedDartTests(unittest.TestCase):
    def test_default_paths_and_external_override_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            repo,python=dart_runtime.installation({},root)
            self.assertEqual(Path(repo),root/'tools/dart')
            self.assertEqual(Path(python),root/'tools/dart/.venv/Scripts/python.exe')
            self.assertEqual(dart_runtime.installation({'dart_repo':'custom','dart_python':'custom-python'},root),('custom','custom-python'))
            with self.assertRaises(ValueError):dart_runtime.installation({'dart_repo':'custom'},root)

    def test_resolution_is_shared_by_dart_artifact_identity(self):
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)):
            values={'detector':'dart','repair':'dart'}
            opts=dust.options(values)
            self.assertEqual(opts['dart_repo'],str(Path(td)/'tools/dart'))
            self.assertEqual(dust.options(opts),opts)
            self.assertEqual(values,{'detector':'dart','repair':'dart'})
            with self.assertRaisesRegex(ValueError,'not set up'):
                dart_runtime.require_installation({},Path(td))

    def test_no_setup_without_explicit_research_acknowledgement(self):
        with tempfile.TemporaryDirectory() as td,patch.object(setup,'run') as commands:
            with self.assertRaises(ValueError):setup.install(Path(td))
            commands.assert_not_called();self.assertFalse((Path(td)/'tools').exists())

    def test_verified_weight_copy_is_read_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td,patch.object(setup,'digest',return_value=setup.DART_WEIGHTS_SHA256),patch.object(setup.urllib.request,'urlopen') as network:
            root=Path(td);repo=root/'managed';source=root/'existing.safetensors';source.write_bytes(b'existing pinned fixture')
            self.assertTrue(setup.install_weights(repo,source,skip=True))
            target=repo/'pretrained_models/dart.safetensors';before=target.stat().st_mtime_ns
            self.assertTrue(setup.install_weights(repo))
            self.assertEqual(target.stat().st_mtime_ns,before)
            self.assertEqual(source.read_bytes(),b'existing pinned fixture');network.assert_not_called()

    def test_bad_existing_or_supplied_weights_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td,patch.object(setup,'digest',return_value='wrong'),patch.object(setup.urllib.request,'urlopen') as network:
            repo=Path(td);source=repo/'old.safetensors';source.write_bytes(b'old')
            with self.assertRaises(ValueError):setup.install_weights(repo,source)
            target=repo/'pretrained_models/dart.safetensors';self.assertFalse(target.exists())
            target.write_bytes(b'user data')
            with self.assertRaises(ValueError):setup.install_weights(repo)
            self.assertEqual(target.read_bytes(),b'user data');network.assert_not_called()

    def test_skip_weights_never_downloads(self):
        with tempfile.TemporaryDirectory() as td,patch.object(setup.urllib.request,'urlopen') as network:
            self.assertFalse(setup.install_weights(Path(td),skip=True));network.assert_not_called()

    def test_download_is_verified_before_publication(self):
        import io
        with tempfile.TemporaryDirectory() as td,patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'bad-download')),patch.object(setup,'digest',return_value='wrong'):
            repo=Path(td)
            with self.assertRaisesRegex(ValueError,'verification'):setup.install_weights(repo)
            self.assertFalse((repo/'pretrained_models/dart.safetensors').exists())

    def test_checkout_requires_clean_exact_pin_without_pull(self):
        with patch.object(setup.subprocess,'check_output',side_effect=[setup.DART_COMMIT,' M tracked.py']),patch.object(setup,'run') as mutate:
            with self.assertRaisesRegex(ValueError,'changes'):setup.checked_checkout(Path('managed'))
            mutate.assert_not_called()
        with patch.object(setup.subprocess,'check_output',return_value='other'):
            with self.assertRaisesRegex(ValueError,'pin'):setup.checked_checkout(Path('managed'))

    def test_installer_has_independent_opt_in_entry_before_comfy_setup(self):
        text=(ROOT/'install_windows.ps1').read_text(encoding='utf-8-sig')
        self.assertLess(text.index('if ($DartOnly)'),text.index('$mode = Resolve-ComfyInstallMode'))
        self.assertIn('AcceptDartResearch',text);self.assertIn('--skip-weights',text)
        self.assertIn('if ($InstallDart)',text)

    def test_cli_bootstraps_without_arp_evaluation_dependencies(self):
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/setup_dart.py'),'--help'],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--acknowledge-research',result.stdout)

    def test_uv_is_reused_only_at_exact_pin_or_installed_in_separate_tooling(self):
        with tempfile.TemporaryDirectory() as td,patch.object(setup.shutil,'which',return_value='uv'), \
             patch.object(setup.subprocess,'check_output',return_value=f'uv {setup.UV_VERSION} (fixture)'), \
             patch.object(setup,'run') as commands:
            self.assertEqual(setup.uv_command(Path(td)),['uv']);commands.assert_not_called()
        with tempfile.TemporaryDirectory() as td,patch.object(setup.shutil,'which',return_value=None), \
             patch.object(setup,'run') as commands:
            root=Path(td);command=setup.uv_command(root)
            self.assertTrue(Path(command[0]).is_relative_to(root/'tools/dart-installer'))
            pip=commands.call_args.args[0]
            self.assertIn('--isolated',pip);self.assertIn('--require-virtualenv',pip)
            self.assertIn(f'uv=={setup.UV_VERSION}',pip)

    def test_explicit_external_paths_keep_previous_candidate_identity(self):
        values={'detector':'dart','repair':'dart','dart_repo':r'I:\DART',
                'dart_python':r'I:\DART\.venv\Scripts\python.exe'}
        opts=dust.options(values)
        self.assertEqual(opts['dart_repo'],values['dart_repo'])
        self.assertEqual(opts['dart_python'],values['dart_python'])
        self.assertEqual(opts['research_confirmed'],'false')
        self.assertEqual(opts['license_confirmed'],'false')

    def test_missing_installation_status_never_runs_a_probe_or_installs(self):
        from types import SimpleNamespace
        app=SimpleNamespace(stabilized_source_for_downstream=lambda:'source.mkv')
        with tempfile.TemporaryDirectory() as td,patch.object(dust,'ROOT',Path(td)), \
             patch.object(dust,'saved_mask',side_effect=FileNotFoundError),patch.object(setup,'run') as commands:
            status=dust.status(app,{'detector':'dart','repair':'dart'})
            self.assertEqual(status['frames'],0)
            self.assertFalse(status['installation']['available'])
            self.assertIn('-DartOnly',status['installation']['message']);commands.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows installer entry')
    def test_powershell_dart_only_dispatch_is_independent_and_forwards_switches(self):
        script=r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $env:ARP_TEST_ROOT 'install_windows.ps1'),[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw 'Invalid PowerShell AST' }
$function=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Install-ManagedDart'},$false)
Invoke-Expression $function.Extent.Text
function Invoke-Step { param($Label,$Block) & $Block }
function Invoke-External { param([string[]]$Command) Write-Output ('DISPATCH:'+($Command|ConvertTo-Json -Compress)) }
$Root=$env:ARP_TEST_ROOT
$PipelinePython=$env:ARP_TEST_PYTHON
$DartOnly=$true; $AcceptDartResearch=$true; $NonInteractive=$true
$SkipModelDownloads=$true; $DartWeightsFrom='I:\source weights\dart.safetensors'
$branch=$ast.Find({param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Extent.Text.StartsWith('if ($DartOnly)')},$false)
Invoke-Expression $branch.Extent.Text
throw 'DartOnly did not exit'
'''
        import json
        result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',script],
                              env={**os.environ,'ARP_TEST_ROOT':str(ROOT),'ARP_TEST_PYTHON':sys.executable},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        dispatch=next(line for line in result.stdout.splitlines() if line.startswith('DISPATCH:'))
        args=json.loads(dispatch.removeprefix('DISPATCH:'))
        self.assertEqual(args[1],str(ROOT/'scripts/setup_dart.py'))
        self.assertIn('--acknowledge-research',args);self.assertIn('--skip-weights',args)
        self.assertEqual(args[-2:],['--weights-from',r'I:\source weights\dart.safetensors'])

    def test_install_and_retry_use_only_managed_runtime_and_frozen_sync(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);repo=root/'tools/dart'
            managed=repo/'.python/cpython/python.exe'
            calls=[]
            def command(args,**kwargs):
                calls.append(([str(arg) for arg in args],kwargs))
                if args[0:2]==['git','clone']:
                    Path(args[-1]).mkdir()
            with patch.object(setup,'run',side_effect=command),patch.object(setup,'checked_checkout'), \
                 patch.object(setup,'uv_command',return_value=['pinned-uv']), \
                 patch.object(setup.subprocess,'check_output',return_value=str(managed)) as find, \
                 patch.object(setup,'install_weights',return_value=True), \
                 patch('dust_model_preflight.inspect_dart',return_value={'backend':'dart','ready':True}) as preflight, \
                 patch.dict(os.environ,{'UV_PROJECT_ENVIRONMENT':'external','UV_PYTHON':'external',
                                        'VIRTUAL_ENV':'external','PYTHONPATH':'external'}):
                setup.install(root,acknowledged=True)
                setup.install(root,acknowledged=True)
            self.assertEqual(sum(args[:2]==['git','clone'] for args,_ in calls),1)
            for args,kwargs in calls:
                if args[0]!='pinned-uv':continue
                env=kwargs['env']
                self.assertEqual(env['UV_PROJECT_ENVIRONMENT'],str(repo/'.venv'))
                self.assertEqual(env['UV_PYTHON_INSTALL_DIR'],str(repo/'.python'))
                self.assertNotIn('VIRTUAL_ENV',env);self.assertNotIn('PYTHONPATH',env);self.assertNotIn('UV_PYTHON',env)
                if 'install' in args:
                    self.assertIn('--no-bin',args);self.assertIn('--no-registry',args)
                if 'sync' in args:
                    self.assertIn('--frozen',args);self.assertIn('--no-dev',args)
                    self.assertEqual(args[args.index('--python')+1],str(managed))
            self.assertEqual(preflight.call_count,2)
            self.assertEqual(find.call_count,2)
            self.assertIn('--system',find.call_args.args[0])
            self.assertIn('--managed-python',find.call_args.args[0])

    @unittest.skipUnless(os.name=='nt','Windows installer prompts')
    def test_normal_installer_offers_dart_but_requires_opt_in_and_research_consent(self):
        import json
        script=r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $env:ARP_TEST_ROOT 'install_windows.ps1'),[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw 'Invalid PowerShell AST' }
foreach ($name in @('Read-Choice','Install-ManagedDart')) {
    $node=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$false)
    Invoke-Expression $node.Extent.Text
}
function Read-Host {
    param($Prompt)
    $script:prompts+=1
    if ($script:answers.Count -eq 0) { throw 'Unexpected interactive prompt' }
    return $script:answers.Dequeue()
}
function Invoke-Step { param($Label,$Block) & $Block }
function Invoke-External { param([string[]]$Command) $script:dispatched=$Command }
$case=$env:ARP_TEST_CASE | ConvertFrom-Json
$Root=$env:ARP_TEST_ROOT
$PipelinePython=$env:ARP_TEST_PYTHON
$InstallDart=[bool]$case.install; $NonInteractive=[bool]$case.noninteractive
$AcceptDartResearch=[bool]$case.accept; $SkipModelDownloads=$false; $DartWeightsFrom=''
$script:answers=New-Object System.Collections.Queue
foreach ($answer in $case.answers) { $script:answers.Enqueue($answer) }
$script:prompts=0; $script:dispatched=@(); $completed=$false; $failure=''
try {
    $selection=$ast.Find({param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Extent.Text.StartsWith('if (-not $InstallDart -and -not $NonInteractive)')},$false)
    Invoke-Expression $selection.Extent.Text
    $dispatch=$ast.Find({param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Extent.Text.StartsWith('if ($InstallDart)')},$false)
    Invoke-Expression $dispatch.Extent.Text
    $completed=$true
} catch { $failure=$_.Exception.Message }
Write-Output ('REPORT:'+(@{prompts=$script:prompts;args=@($script:dispatched);completed=$completed;error=$failure}|ConvertTo-Json -Compress))
'''
        cases=[
            ('enter skips',{},[''],False,True),
            ('no skips',{},['n'],False,True),
            ('yes and consent',{},['y','yes'],True,True),
            ('declined consent',{},['y','n'],False,True),
            ('empty consent',{},['y',''],False,True),
            ('invalid choice retries',{},['invalid','Y','y'],True,True),
            ('explicit component',{'install':True},['yes'],True,True),
            ('acknowledgement does not select',{'accept':True},['n'],False,True),
            ('preacknowledged interactive',{'accept':True},['y'],True,True),
            ('noninteractive default',{'noninteractive':True},[],False,True),
            ('noninteractive acknowledgement alone',{'noninteractive':True,'accept':True},[],False,True),
            ('noninteractive explicit',{'noninteractive':True,'install':True,'accept':True},[],True,True),
            ('noninteractive missing consent',{'noninteractive':True,'install':True},[],False,False),
        ]
        for name,flags,answers,installs,completed in cases:
            with self.subTest(name=name):
                case={**flags,'answers':answers}
                result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',script],
                    env={**os.environ,'ARP_TEST_ROOT':str(ROOT),'ARP_TEST_PYTHON':sys.executable,'ARP_TEST_CASE':json.dumps(case)},capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads(next(line.removeprefix('REPORT:') for line in result.stdout.splitlines() if line.startswith('REPORT:')))
                self.assertEqual(report['prompts'],len(answers),report)
                self.assertEqual(report['completed'],completed,report)
                if installs:
                    self.assertEqual(report['args'][1],str(ROOT/'scripts/setup_dart.py'))
                    self.assertIn('--acknowledge-research',report['args'])
                else:self.assertEqual(report['args'],[])
                if not completed:self.assertIn('-AcceptDartResearch',report['error'])

    def test_failed_sync_can_retry_without_cloning_or_publishing_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);repo=root/'tools/dart';repo.mkdir(parents=True)
            managed=repo/'.python/python.exe'
            with patch.object(setup,'checked_checkout'),patch.object(setup,'uv_command',return_value=['uv']), \
                 patch.object(setup.subprocess,'check_output',return_value=str(managed)), \
                 patch.object(setup,'install_weights',return_value=False) as weights:
                with patch.object(setup,'run',side_effect=[None,subprocess.CalledProcessError(1,'sync')]):
                    with self.assertRaises(subprocess.CalledProcessError):setup.install(root,acknowledged=True)
                weights.assert_not_called()
                with patch.object(setup,'run') as commands:
                    setup.install(root,acknowledged=True,skip_weights=True)
                self.assertEqual(commands.call_count,2)

    @unittest.skipUnless(os.name=='nt','Windows junction containment')
    def test_external_junctions_are_rejected_before_setup_or_weight_write(self):
        for redirect in ('tools','tools/dart','tools/dart/.venv','tools/dart/.python','weights'):
            with self.subTest(redirect=redirect),tempfile.TemporaryDirectory() as td:
                root=Path(td)/'arp';root.mkdir();external=Path(td)/'external';external.mkdir()
                marker=external/'keep';marker.write_bytes(b'untouched')
                link=root/('tools/dart/pretrained_models' if redirect=='weights' else redirect)
                link.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run(['cmd','/c','mklink','/J',str(link),str(external)],check=True,capture_output=True)
                with patch.object(setup,'run') as commands,patch.object(setup,'uv_command') as uv:
                    with self.assertRaisesRegex(ValueError,'external|outside'):
                        if redirect=='weights':setup.install_weights(root/'tools/dart')
                        else:setup.install(root,acknowledged=True)
                    commands.assert_not_called();uv.assert_not_called()
                self.assertEqual(marker.read_bytes(),b'untouched')


if __name__=='__main__':unittest.main()

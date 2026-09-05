"""Contract-bound ACT, raw-PACT and finetuned PACT-readout evaluator."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pact_workflow import digest, file_digest, load_contract, resolve, write_json
from pact import verify_runtime
from pact_checkpoint import paired_encoder_checkpoint
from pact_eval_protocol import rollout, summarize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--checkpoint-name', default='policy_best.ckpt')
    p.add_argument('--suite', choices=('smoke', 'dev', 'test'), default='smoke')
    p.add_argument('--reference', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--worker', type=int)
    p.add_argument('--result', type=Path)
    p.add_argument('--verify-horizon', type=int)
    return p.parse_args()


def identity(args, contract):
    checkpoint = args.checkpoint_dir / args.checkpoint_name
    if Path(args.checkpoint_name).name != args.checkpoint_name:
        raise ValueError('checkpoint-name must be a filename')
    paths = [checkpoint, args.checkpoint_dir / 'dataset_stats.pkl']
    paths += [p for p in (args.checkpoint_dir / 'prox_config.json',
                          args.checkpoint_dir / 'training_config.json') if p.exists()]
    prox_path = args.checkpoint_dir / 'prox_config.json'
    if prox_path.exists():
        encoder = paired_encoder_checkpoint(args.checkpoint_dir, json.loads(prox_path.read_text()), args.checkpoint_name)
        if encoder is not None:
            paths.append(encoder)
    code = [Path(__file__), ROOT / 'scripts/pact_eval_protocol.py', ROOT / 'scripts/pact_workflow.py',
            ROOT / 'scripts/pact.py', ROOT / 'scripts/pact_checkpoint.py', Path(__file__).with_name('eval_place_fast_hooks.py'),
            Path(__file__).with_name('eval_act_obstacle.py'), Path(__file__).with_name('policy.py')]
    code += sorted((ROOT / 'encoders').rglob('*.py'))
    code += sorted((ROOT / 'submodules/act/detr').rglob('*.py'))
    code += [Path(__file__).with_name('prox_cvae.py')]
    scenes = sorted((ROOT / 'custom_scenes').glob('*.xml'))
    versions = {}
    for name in ('torch', 'torchvision', 'mujoco', 'warp-lang', 'numpy', 'scipy'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    payload = {'contract': contract['sha256'],
               'checkpoint_files': {p.name: file_digest(p) for p in paths},
               'code': {str(p.relative_to(ROOT)): file_digest(p) for p in code},
               'scenes': {p.name: file_digest(p) for p in scenes},
               'runtime': file_digest(resolve(contract['profile']['runtime_dir']) / 'runtime.json'),
               'protocol': 'ever_success_full_horizon_v2', 'proximity': 'native_egl_substeps_min',
               'geometry_history': 'consecutive_control_steps_v1',
               'python': sys.version, 'platform': sys.platform, 'package_versions': versions}
    payload['sha256'] = digest(payload)
    return payload


def configure_proximity(pc, directory, policy_name, contract):
    path = directory / 'prox_config.json'
    if not path.exists():
        return
    prox = json.loads(path.read_text())
    feature = prox.get('prox_feature')
    if feature not in ('raw', 'peak_closeness'):
        expected = {'prox_feature': 'surface_embedding', 'finetune_prox_encoder': True,
                    'prox_policy_tap': 'readout', 'prox_feat_dim': 128,
                    'n_proximity_sensors': 40, 'prox_tokens_per_sensor': 1,
                    'prox_layout': 'per_sensor', 'proximity_layout': 'raw_causal'}
        for key, value in expected.items():
            if prox.get(key) != value:
                raise ValueError(f'Unsupported readout configuration: {key} must be {value!r}')
        pc.prox_encoder_ckpt = str(paired_encoder_checkpoint(directory, prox, policy_name))
        pc.finetune_prox_encoder = True
        pc.prox_policy_tap = 'readout'
    if prox.get('prox_pool') != contract['prox_pool']:
        raise ValueError('Checkpoint proximity pooling differs from data')
    pc.use_proximity = True
    for key in ('prox_feature', 'prox_layout', 'prox_pool', 'prox_tokens_per_sensor'):
        setattr(pc, key, prox[key])


def configure(args, contract, row):
    from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
    from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import FrankaSkinPACTCollisionCorridorConfig
    from molmo_spaces.tasks import enclosure_reach
    from eval_act_obstacle import ACTPolicyConfig
    from eval_place_v1010_scene import V1010_SCENE_BY_POSE, assert_v1010_scene_hashes
    profile = contract['profile']
    cfg = FrankaSkinPACTCollisionCorridorConfig(output_dir=args.result.parent, num_workers=1)
    cfg.task_type = 'pick_and_place'
    cfg.task_config = PickAndPlaceTaskConfig(task_cls=enclosure_reach.PactPlaceCorridorTask)
    cfg.task_horizon = args.verify_horizon or profile['horizon']
    cfg.policy_dt_ms = profile['policy_dt_ms']
    cfg.end_on_success = False
    if hasattr(cfg, 'terminate_upon_success'):
        cfg.terminate_upon_success = False
    if profile['adapter'] in ('v1011d', 'v12'):
        cfg.camera_config = FrankaSkinHybridCameraSystem()
    cfg.proximity_sensor_period_ms = 16.6667
    cfg.viz_sensor_rgb = False
    for name in ('save_videos', 'use_wandb', 'use_passive_viewer', 'filter_for_successful_trajectories'):
        if hasattr(cfg, name):
            setattr(cfg, name, False)
    cfg.robot_config.action_noise_config.enabled = False
    cfg.task_sampler_config.task_sampler_class = getattr(enclosure_reach, profile['sampler_class'])
    pinned_scenes = resolve(profile['runtime_dir']) / 'molmo_spaces/data_generation/custom_scenes'
    if profile['adapter'] in ('v1011d', 'v12'):
        for filename in ('pact_place_corridor_v5.xml', 'pact_place_corridor_v3.xml'):
            if file_digest(ROOT / 'custom_scenes' / filename) != file_digest(pinned_scenes / filename):
                raise ValueError(f'Scene include differs from pinned runtime: {filename}')
        assert_v1010_scene_hashes(ROOT / 'custom_scenes')
        scene = ROOT / 'custom_scenes' / V1010_SCENE_BY_POSE[row['pose_id']]['filename']
        if profile['adapter'] == 'v12':
            scene = pinned_scenes / profile['scene_filename']
            if file_digest(scene) != profile['scene_sha256'] or row['pact_v106_scene_sha256'] != profile['scene_sha256']:
                raise ValueError('v12 preview scene differs from collection')
    else:
        scene = pinned_scenes / 'pact_place_corridor_v2.xml'
    cfg.task_sampler_config.scene_xml_paths = [str(scene)] * 2
    pc = ACTPolicyConfig(ckpt_dir=str(args.checkpoint_dir), ckpt_name=args.checkpoint_name,
                         camera_names=tuple(profile['camera_names']), chunk_size=profile['chunk_size'],
                         image_h=contract['image_h'], image_w=contract['image_w'], temp_agg_off=True)
    import torch
    weights = torch.load(args.checkpoint_dir / args.checkpoint_name, map_location='cpu', weights_only=True)
    checkpoint_chunk = int(weights['model.query_embed.weight'].shape[0])
    pc.chunk_size = checkpoint_chunk
    del weights
    training = args.checkpoint_dir / 'training_config.json'
    if training.exists():
        saved = json.loads(training.read_text())
        if saved['experiment_sha256'] != contract['sha256']:
            raise ValueError('Checkpoint was trained with another experiment contract')
        policy_config = saved['policy_config']
        if policy_config['camera_names'] != profile['camera_names']:
            raise ValueError('Checkpoint camera order differs from experiment')
        for name in ('hidden_dim', 'dim_feedforward', 'enc_layers', 'dec_layers', 'nheads', 'backbone', 'state_dim', 'action_dim'):
            if name in policy_config:
                setattr(pc, name, policy_config[name])
        pc.chunk_size = policy_config['num_queries']
        if pc.chunk_size != checkpoint_chunk:
            raise ValueError('Checkpoint chunk shape differs from saved architecture')
    configure_proximity(pc, args.checkpoint_dir, args.checkpoint_name, contract)
    # Keep policy camera geometry/resolution unchanged from collection. Reference
    # also keeps annotations and unused depth for comparison with the original.
    if not args.reference:
        cfg.camera_config.cameras = [c for c in cfg.camera_config.cameras
                                     if c.name in pc.camera_names or (pc.use_proximity and getattr(c, 'is_proximity_sensor', False))]
    cfg.policy_config = pc
    return cfg


def worker(args, contract, row, run_identity):
    runtime = verify_runtime(contract['profile'])
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('MLSPACES_ASSETS_DIR', str(ROOT / 'assets'))
    os.environ.setdefault('PACT_CONTACT_AUDIT_SUMMARY_ONLY', '1')
    sys.path.insert(0, str(runtime))
    sys.path.insert(1, str(ROOT))
    import numpy as np
    import torch
    from utils import set_seed
    from eval_act_obstacle import ACTInferencePolicy
    from eval_place_fast_hooks import _install_metrics_only_sensor_filter, _install_contract_sensor_gate
    from molmo_spaces.tasks import pact_place_contact_audit as contact_module
    from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit, place_environment_contact_pairs
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    assert_supported_runtime(strict=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.reference:
        _install_metrics_only_sensor_filter()
        _install_contract_sensor_gate()
    cfg = configure(args, contract, row)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    record = {'status': 'error', 'row_sha256': row['sha256'], 'identity': run_identity['sha256'],
              'reference': args.reference, 'row': row, 'runtime_verification': bool(args.verify_horizon)}
    started = time.perf_counter()
    try:
        v12_overlay = None
        if contract['profile']['adapter'] == 'v12':
            from pact_v12_adapter import install_sampler_overlay, apply_overlay
            v12_overlay = install_sampler_overlay(sampler, runtime)
        # No policy-dependent retries or replacement of failed episodes. A failed
        # scene is an explicit infrastructure failure; fix the suite deliberately.
        set_seed(row['task_seed_u32'])
        sampler.seed_task_sampling(row['task_seed_u32'])
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=row['scene_template_house_index'])
        if task is None:
            raise ValueError('Sampler returned no task')
        if v12_overlay is not None:
            record['overlay'] = apply_overlay(task, *v12_overlay)
        params = task.scene_params
        if params.get('pact_place_environment_version') != contract['profile']['environment_version']:
            raise ValueError('Realized environment differs from experiment')
        if params.get('pact_v1011_layout_sha256') in {e['layout_sha256'] for e in contract['episodes'] if e['layout_sha256']}:
            raise ValueError('Evaluation world repeats a training/validation world')
        initial_contacts = list(place_environment_contact_pairs(task.env))
        forbidden = {'hazard_bar', 'other_environment', 'clutter', 'mounted_fixture'}
        if any(contact_module.classify_contact(c) in forbidden for c in initial_contacts):
            raise ValueError('Scene has forbidden robot/environment contact before the policy starts')
        policy = ACTInferencePolicy(cfg, task)
        policy.prepare_model()
        for prefix, width in (('qpos', 9), ('action', 8)):
            for suffix in ('mean', 'std'):
                values = np.asarray(policy._stats[f'{prefix}_{suffix}'])
                if values.shape != (width,) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid checkpoint normalization: {prefix}_{suffix}')
                if suffix == 'std' and np.any(values <= 0):
                    raise ValueError('Normalization standard deviations must be positive')
        task.register_policy(policy)
        if policy._prox_encoder is not None:
            prox = json.loads((args.checkpoint_dir / 'prox_config.json').read_text())
            if list(policy._prox_encoder.sensor_order) != prox['sensor_order']:
                raise ValueError('Live encoder sensor order differs from checkpoint')
        audit = PactPlaceContactAudit()
        task._contact_audit_hook = audit
        # A clean reset must not consume a depth buffer left by scene setup.
        task.env.reset_proximity_depth_buffer(task._proximity_camera_names)
        traces, inputs = [], []
        original_inference = policy.inference_model
        def inference(obs):
            if policy.needs_fresh_policy_observation() and args.verify_horizon:
                keys = ['qpos']
                if policy.needs_fresh_camera_observation():
                    keys += list(cfg.policy_config.camera_names)
                if policy.needs_fresh_proximity_observation():
                    keys += list(policy._prox_encoder.sensor_order)
                h = hashlib.sha256()
                def add(value):
                    if isinstance(value, dict):
                        for k in sorted(value):
                            h.update(k.encode()); add(value[k])
                    else:
                        a = np.asarray(value)
                        h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
                for key in keys:
                    add(obs[key])
                inputs.append(h.hexdigest())
            return original_inference(obs)
        policy.inference_model = inference
        def trace(step, action, observation, success):
            if args.verify_horizon:
                qpos = observation[0]['qpos']
                traces.append({'step': step, 'arm': np.asarray(action['arm']).tolist(),
                               'gripper': np.asarray(action['gripper']).tolist(),
                               'qpos': np.concatenate([qpos['arm'], qpos['gripper']]).tolist(),
                               'success': success})
        with torch.inference_mode():
            record.update(rollout(task, policy, cfg.task_horizon, trace=trace))
        contact = audit.summary()
        record.update(collision_free=bool(contact['collision_free']), contact_audit=contact,
                      realized_layout_sha256=params.get('pact_v1011_layout_sha256'),
                      gripper_close_commanded=policy.gripper_close_commanded,
                      trace=traces, policy_input_hashes=inputs,
                      torch_version=torch.__version__)
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        record['wall_seconds'] = time.perf_counter() - started
        write_json(args.result, record)
        sampler.close()
    return record


def compare(reference, optimized):
    import numpy as np
    if reference['status'] != 'complete' or optimized['status'] != 'complete':
        return False
    if reference['policy_input_hashes'] != optimized['policy_input_hashes']:
        return False
    if not reference['policy_input_hashes'] or len(reference['trace']) != len(optimized['trace']):
        return False
    for a, b in zip(reference['trace'], optimized['trace']):
        if a['success'] != b['success']:
            return False
        for key in ('arm', 'gripper', 'qpos'):
            if not np.allclose(a[key], b[key], rtol=0, atol=1e-6):
                return False
    return (reference['success'] == optimized['success'] and
            reference['terminal_success'] == optimized['terminal_success'] and
            reference['contact_audit'] == optimized['contact_audit'])


def main():
    args = parse_args()
    contract = load_contract(args.run_dir / 'experiment.json')
    run_identity = identity(args, contract)
    rows = contract['evaluation']['test' if args.suite == 'test' else 'dev']
    if args.suite == 'smoke':
        rows = rows[:2]
    if args.worker is not None:
        record = worker(args, contract, rows[args.worker], run_identity)
        return 0 if record['status'] == 'complete' else 1
    output = args.run_dir / 'evaluation' / run_identity['sha256'][:16]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'identity.json', run_identity)
    verification = output / 'verification.json'
    if args.suite == 'test' and not args.reference:
        if not verification.exists() or not json.loads(verification.read_text()).get('passed'):
            raise ValueError('Run scripts/pact.py verify --run NAME first; optimized test needs matching trace parity evidence')
    def execute(index, reference, label, horizon=None):
        result_path = output / label / f'{index:03d}.json'
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get('identity') != run_identity['sha256'] or existing.get('row_sha256') != rows[index]['sha256']:
                raise ValueError(f'Resume identity mismatch: {result_path}')
            if existing.get('status') == 'complete':
                print(f'Resume completed row: {result_path}', flush=True)
                return existing
        result_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
                   '--checkpoint-dir', str(args.checkpoint_dir), '--checkpoint-name', args.checkpoint_name,
                   '--suite', args.suite, '--worker', str(index), '--result', str(result_path)]
        if reference:
            command += ['--reference']
        if horizon:
            command += ['--verify-horizon', str(horizon)]
        print(f'Running {label} row {index + 1}/{len(rows)}; log={result_path.with_suffix(".log")}', flush=True)
        with result_path.with_suffix('.log').open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if not result_path.exists():
            write_json(result_path, {'status': 'error', 'identity': run_identity['sha256'],
                                    'row_sha256': rows[index]['sha256'], 'error': f'Worker exit {proc.returncode}; see log'})
        record = json.loads(result_path.read_text())
        if proc.returncode and record.get('status') == 'complete':
            record.update(status='error', error=f'Worker exited {proc.returncode} after writing results; inspect log')
            write_json(result_path, record)
        print(f"{label} row {index + 1}: {record['status']}, success={record.get('success')}, "
              f"seconds={record.get('wall_seconds')}, error={record.get('error')}", flush=True)
        return record
    if args.verify:
        # At least two complete chunks and the following query. This is a bounded
        # parity regression check, not an expert-solvability or judge calibration.
        import torch
        weights = torch.load(args.checkpoint_dir / args.checkpoint_name, map_location='cpu', weights_only=True)
        chunk = int(weights['model.query_embed.weight'].shape[0])
        del weights
        horizon = min(contract['profile']['horizon'], 2 * chunk + 1)
        comparisons = []
        for i in range(min(2, len(rows))):
            reference = execute(i, True, 'verify_reference', horizon)
            optimized = execute(i, False, 'verify_optimized', horizon)
            comparisons.append(compare(reference, optimized))
        report = {'passed': all(comparisons), 'identity': run_identity['sha256'],
                  'rows': comparisons, 'horizon': horizon, 'scope': 'observation/action/state/contact parity; task solvability not certified'}
        write_json(verification, report)
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    records = []
    started = time.perf_counter()
    for i in range(len(rows)):
        records.append(execute(i, args.reference, args.suite + ('_reference' if args.reference else '')))
        report = summarize(records, len(rows))
        report.update(identity=run_identity['sha256'], suite=args.suite,
                      dataset=contract['dataset'],
                      evaluation_variant=contract['profile'].get('dataset_environment_version', contract['profile']['environment_version']),
                      sampler_class=contract['profile']['sampler_class'],
                      proximity_backend='native_egl_substeps_min', task_horizon=contract['profile']['horizon'],
                      reference=args.reference, success_definition='ever_success',
                      collision_window='full_horizon', session_wall_seconds=time.perf_counter() - started)
        write_json(output / (args.suite + ('_reference' if args.reference else '') + '.json'), report)
        if records[-1]['status'] != 'complete':
            print(json.dumps(report, indent=2))
            return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

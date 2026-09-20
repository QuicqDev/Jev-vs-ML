"""Immutable run manifests, shared holdouts, and two isolated training lanes."""
from .common import *
from .datasets import prepare_data, make_holdout, make_split
from .training import train_lane, finish_result
from .api import JevClient, get_key, make_payload, few_shot_ids


def prepare_suite(root, cfg, pilot_root=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    manifest = dict(configuration=cfg, code_hashes=sources, environment=environment())
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if digest(prior) != digest(manifest):
            raise ValueError('Configuration, code, or environment changed: choose a new output ROOT to avoid mixed results.')
    else:
        write_json(manifest_path, manifest)
    rows = []
    for name in tqdm(cfg['datasets'], desc='Prepare datasets'):
        slug = name.replace(' ', '_')
        if pilot_root:
            source = Path(pilot_root) / 'data' / slug
            target = root / 'data' / slug
            if source.exists() and not target.exists():
                if cfg['max_text_chars'] != 4000:
                    raise ValueError('Pilot snapshot reuse requires original 4000-character truncation')
                target.parent.mkdir(exist_ok=True)
                shutil.copytree(source, target)
        df, meta = prepare_data(name, root, cfg)
        holdout = make_holdout(df, cfg)
        write_json(root / 'data' / slug / 'holdout.json', {k: v.tolist() for k, v in holdout.items()})
        for seed in cfg['seeds']:
            split = make_split(df, holdout, cfg, seed)
            write_json(root / 'runs' / slug / str(seed) / 'split.json', {k: v.tolist() for k, v in split.items()})
            rows.append(dict(dataset=name, seed=seed, classes=len(meta['labels']),
                **{k: len(v) for k, v in split.items()},
                test_class_counts={str(k): int(v) for k, v in df.iloc[split['test']].label.value_counts().sort_index().items()},
                excluded_pilot_test_rows=len(holdout['pilot_excluded'])))
    overview = pd.DataFrame(rows)
    overview.to_csv(root / 'dataset_overview.csv', index=False)
    calls = int(sum((r['test'] + (r['policy'] if r['classes'] == 2 else 0)) * len(cfg['jev_modes']) for r in rows))
    print(f'Jev upper bound before cache/retries: {calls:,} requests; retry ceiling: {calls * (1 + cfg["max_retries"]):,}.')
    print('The cost guard is an input-token estimate, not a billing cap. Verify current API pricing before running Jev.')
    return overview


def run_ml(root, cfg, gpu_ids=None):
    if cfg.get('use_cuml'):
        from .gpu_process import run_isolated
        return run_isolated(root, cfg, gpu_ids)
    from joblib import Parallel, delayed, parallel_config
    if gpu_ids is None:
        gpu_ids, device_status = probe_gpus()
        print(device_status)
    if cfg.get('require_gpu') and not gpu_ids:
        raise RuntimeError('No GPU passed both library probes. Enable Kaggle GPU and inspect probe errors above. '
                           'CPU fallback is disabled for this preset.')
    jobs = [(d, s) for d in cfg['datasets'] for s in cfg['seeds']]
    workers = min(len(jobs), cfg['max_parallel_jobs'], len(gpu_ids) if gpu_ids else cfg['max_parallel_jobs'])
    lanes = [jobs[i::workers] for i in range(workers)]
    if workers == 1:
        return train_lane(lanes[0], root, cfg, gpu_ids[0] if gpu_ids else None)
    with parallel_config(backend='loky', inner_max_num_threads=cfg['threads']):
        results = Parallel(n_jobs=workers, verbose=10)(
            delayed(train_lane)(lane, root, cfg, gpu_ids[i] if gpu_ids else None) for i, lane in enumerate(lanes))
    return [r for lane in results for r in lane]


def request_partition(client, df, ids, meta, examples, cfg, description):
    def one(idx):
        return client.call(make_payload(df, idx, meta, examples, cfg))
    with futures.ThreadPoolExecutor(max_workers=cfg['jev_workers']) as pool:
        responses = list(tqdm(pool.map(one, ids), total=len(ids), desc=description))
    p = np.array([r['probabilities'] if r['ok'] else [np.nan] * len(meta['labels']) for r in responses])
    return dict(raw=np.array([r['prediction'] for r in responses]),
                score=p[:, 1] if len(meta['labels']) == 2 else p,
                probability=p, default_threshold=.5), responses


def run_jev(root, cfg, client=None):
    root = Path(root)
    client = client or JevClient(root, cfg, get_key())
    for dataset in cfg['datasets']:
        slug = dataset.replace(' ', '_')
        df = pd.read_parquet(root / 'data' / slug / 'snapshot.parquet')
        meta = json.loads((root / 'data' / slug / 'metadata.json').read_text())
        classes = len(meta['labels'])
        for seed in cfg['seeds']:
            folder = root / 'runs' / slug / str(seed)
            split = json.loads((folder / 'split.json').read_text())
            for mode in cfg['jev_modes']:
                name = 'Jev ' + mode
                result_path = folder / (name + '.json')
                if result_path.exists():
                    continue
                examples = few_shot_ids(df, split, cfg, seed) if mode == 'few-shot' else []
                policy_responses = []
                if classes == 2:
                    policy, policy_responses = request_partition(client, df, split['policy'], meta, examples, cfg,
                        f'{dataset} {seed} {mode}: policy')
                else:
                    policy = dict(raw=np.array([], dtype=int), score=np.array([]), probability=None, default_threshold=.5)
                test, responses = request_partition(client, df, split['test'], meta, examples, cfg,
                    f'{dataset} {seed} {mode}: test')
                write_json(folder / (name + '.responses.json'), dict(policy=policy_responses, test=responses))
                result = finish_result(name, df.iloc[split['policy']].label.to_numpy(), df.iloc[split['test']].label.to_numpy(),
                    policy, test, classes, cfg, dict(examples=examples, requested_model=cfg['jev_model'],
                        resolved_models=sorted({str(r.get('resolved_model')) for r in responses + policy_responses if r['ok']}),
                        failed_test_rows=sum(not r['ok'] for r in responses),
                        failed_policy_rows=sum(not r['ok'] for r in policy_responses),
                        cache_hits=sum(r['cache_hit'] for r in responses + policy_responses)))
                result.update(dataset=dataset, seed=seed, test_ids=split['test'], test_labels=df.iloc[split['test']].label.tolist())
                write_json(result_path, result)
    return dict(attempts=getattr(client, 'attempts', None), estimated_input_cost=getattr(client, 'estimated_cost', None))

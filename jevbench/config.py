"""Versioned experiment policy; no hidden per-dataset score-driven tuning."""
from .common import ALL_DATASETS


def configuration(preset='benchmark'):
    if preset == 'v3':
        cfg = configuration('benchmark')
        cfg.update(protocol='3.0.1', preset='v3', speed_profile=True, use_cuml=True,
                   require_gpu=True, train_cap=8000, validation_cap=1000,
                   trees=150, histogram_iterations=60, early_stopping=15,
                   word_features=20000, char_features=10000, selected_features=512,
                   svd_components=32, jev_workers=8)
        return cfg
    if preset not in ['quick', 'benchmark']:
        raise ValueError('preset must be quick, benchmark or v3')
    quick = preset == 'quick'
    return dict(protocol='2.0.0', preset=preset, datasets=list(ALL_DATASETS),
                seeds=[2027] if quick else [2027, 2028, 2029], holdout_seed=20260920,
                train_cap=1500 if quick else 12000, validation_cap=300 if quick else 1500,
                policy_cap=150 if quick else 500, test_cap=100 if quick else 1000,
                banking_test_cap=154 if quick else 1500,
                exclude_pilot_tests=True, pilot_seeds=[42, 43, 44], pilot_test_cap=300,
                max_trials=2 if quick else 4, trees=60 if quick else 400, early_stopping=30,
                threads=2, max_parallel_jobs=2, word_features=10000 if quick else 30000,
                char_features=5000 if quick else 20000, selected_features=500 if quick else 2000,
                svd_components=64, max_text_chars=4000,
                threshold_quantiles=101, jev_model='jev-1.13.0',
                jev_modes=['zero-shot', 'few-shot'], examples_per_class=1,
                jev_workers=4, min_request_interval=.15, max_api_attempts=65000,
                max_retries=2, estimated_usd_per_million_input_tokens=.042,
                max_estimated_api_usd=20.0, bootstrap_samples=1000)

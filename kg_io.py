{
  "checkpoint": "rotate_path_d64.npz",
  "checkpoint_sha256": "3f4ec625b63a2bd5fb97558ec666ec98bf6b5684cf5a59e3ceeb30135a460c38",
  "dataset_name": "FB15k-237",
  "dataset_url": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/",
  "dataset_file_sha256": {
    "train.txt": "61099230e4439f90885ca9767739e31e8e32f54736fa1c35952b27997bc7c08a",
    "valid.txt": "749cbe9d923bac7b9354da5614ecfed2e0220256d442c3e04a6b303db1f273d9",
    "test.txt": "e2e35e8e6113de220140b6f44dc71a5207b0fc6872d575e874aefe13259b655b"
  },
  "n_entities": 14541,
  "n_relations": 237,
  "dim": 64,
  "seed": 0,
  "train_config": {
    "objective": "atomic + path-consistency",
    "warmup_epochs_atomic": 24,
    "joint_epochs": 19,
    "lr": 0.005,
    "batch_size": 2048,
    "negatives": 16,
    "n_paths": 400000,
    "note": "warm-started from atomic-only at ep24"
  },
  "eval_command": "python -m bridge_hrr_v5.cli eval --data-dir DATA --checkpoint rotate_path_d64.npz --pure-checkpoint rotate_pure_d64.npz --n 2500",
  "result": {
    "n_sampled": 2500,
    "report": {
      "leaky_fraction": 0.5168,
      "all": {
        "n": 2500,
        "pop_exact": 0.186,
        "path_mrr": 0.35271547464531866,
        "path_exact": 0.2816,
        "pure_mrr": 0.1779950840536887
      },
      "strict_clean": {
        "n": 1208,
        "pop_exact": 0.08195364238410596,
        "path_mrr": 0.10517639274322078,
        "path_exact": 0.052152317880794705,
        "pure_mrr": 0.021633702991183047
      },
      "leaky": {
        "n": 1292,
        "pop_exact": 0.28328173374613,
        "path_mrr": 0.5841606843494473,
        "path_exact": 0.49613003095975233,
        "pure_mrr": 0.3241905548923163
      },
      "degree_sweep_strict": {
        "overall": {
          "n": 1208,
          "path_mrr": 0.10517639274322078,
          "path_exact": 0.052152317880794705,
          "pop_exact": 0.08195364238410596,
          "pure_mrr": 0.021633702991183047
        },
        "low_r2deg": {
          "n": 412,
          "ans_size": 69.35679611650485,
          "pop_exact": 0.13349514563106796,
          "path_mrr": 0.14237047771379874,
          "path_exact": 0.08737864077669903,
          "pure_mrr": 0.02328200717470613
        },
        "mid_r2deg": {
          "n": 393,
          "ans_size": 170.92111959287533,
          "pop_exact": 0.0916030534351145,
          "path_mrr": 0.08508877441386074,
          "path_exact": 0.030534351145038167,
          "pure_mrr": 0.015209711832003111
        },
        "high_r2deg": {
          "n": 403,
          "ans_size": 95.40694789081886,
          "pop_exact": 0.019851116625310174,
          "path_mrr": 0.08674083690093885,
          "path_exact": 0.03722084367245657,
          "pure_mrr": 0.026213174956310113
        }
      }
    }
  }
}
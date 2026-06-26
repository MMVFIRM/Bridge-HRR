{
  "checkpoint": "rotate_pure_d64.npz",
  "checkpoint_sha256": "6150d38fbcb56e23bc272d7c6a874013d7b94349ab4075ec00c09bfacd26dc95",
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
    "objective": "atomic only (RotatE link prediction)",
    "epochs": 24,
    "lr": 0.005,
    "batch_size": 2048,
    "negatives": 16,
    "note": "atomic-only control; no path-consistency"
  },
  "eval_command": "(used as --pure-checkpoint control)",
  "result": null
}
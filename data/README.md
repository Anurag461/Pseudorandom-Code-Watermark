# Data

`prompts.jsonl` contains 500 C4 RealNewsLike evaluation prefixes, each with 50 token IDs. `human_reference.jsonl` contains the 4,483 disjoint reference continuations used to calibrate the short substitution experiment.

`keys/` contains the 19 fixed-sweep keys and four online keys. Each key file contains its vocabulary partition and construction-specific keys. Model checkpoint identities are in `models.json`; checkpoint weights are stored outside the repository.

The fixed n=4,096, eta=0.05, r=4,055 key for the long substitution experiment is not included in this collection.

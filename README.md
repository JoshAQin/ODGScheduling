# Feedback-Guided Iterative Scheduling for High-Level Synthesis with Operation Dependency Graph

To run the program, please follow the instructions below.

```bash
# Generate synthetic DFGs used for feedback-guided training
$ python3 dfg_generator.py

# Conduct feedback-guided training
$ python3 rl.py --mode -1

# Conduct iterative scheduling with ODG
#   (1) Multi-DFG scheduling (Benchmark DFGs in our paper)
$ python3 rl.py --mode 0 --policy_network {pre-trained_network}
#   (2) Single-DFG scheduling
$ python3 rl.py --mode {test_file_idx} --policy_network {pre-trained_network}
```

To schedule other DFGs:
(1) Name the DFG as `test_dag_{test_file_idx}.dot`.
(2) Modify the file according to example DFGs.
(3) Put the DFG in the `Test` folder.

Other hyperparameters can be adjust accordingly in `rl.py`.

## Requirements
* Python 3.9
* Numpy v1.22
* Pytorch v1.10
* Torch-geometric v2.2.0
* PuLP v2.6
* Matplotlib v3.4
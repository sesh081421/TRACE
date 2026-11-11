# TRACE model preset

[`final_trace_args.txt`](final_trace_args.txt) lists the model and optimizer
options used by the [training example](../README.md#train). Set the
dataset, data root, result directory, seed, and evaluation options on the
command line. The default sweep uses at most 100 epochs, batches of up to 128
positive pairs, patience 10, and seeds 8, 18, 28, 38, and 48. See the
[usage guide](../docs/REPRODUCIBILITY.md) for training and summary commands.

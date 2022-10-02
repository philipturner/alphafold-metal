# AlphaFold ported to the M1 GPU

This fork ports AlphaFold to Swift using Metal Performance Shaders Graph. Although it was [ported to PyTorch](https://github.com/lucidrains/alphafold2) (which uses MPSGraph internally), this fork can utilize MPSGraph's MLIR compilation capabilities. That greatly optimizes it compared to a purely eager implementation, especially since the model is extremely complex and time-recurrent. Furthermore, writing it in Swift provides full access the low-level Metal API. Metal permits more intense optimizations that aren't possible with the more user-friendly PyTorch tensors.

The official AlphaFold repository uses JAX, which graph-compiles code with XLA for optimal performance on Nvidia GPUs. Here, I have optimized it for Apple GPUs so that I can use it on my MacBook Pro for nanotechnology research. These intense optimizations could greatly speedup my workflow - searching for proteins that may be used for transistors or memory cells. Such proteins have never been produced in living organisms, but genetically engineered bacteria can mass-produce them for real-world hardware. Since I don't know what the proteins will be, I need to rapidly search a vast solution space, with minimal latency and no restraints like Colab usage limits.

- Goal: Port the entire framework in a single day using PythonKit and MPSGraph, using https://github.com/lucidrains/alphafold2 as a reference.
- Afterward, compare performance to the Colab notebook that uses an Nvidia K80. Prove it runs faster on my M1 Max, then ridiculously optimize it for inference performance.

## Day 0

Download the reduced dataset, which required 600 GB\* of disk space. Download `aria2` on Homebrew. Change all instances of `--parents` in the download scripts to `-p`.

> \*This comes in several modules of similar sizes. When you need some extra disk space, remove `uniclust30` which consumes ~90 GB. Later, re-run only the script that downloads this dataset. Freeing up this dataset provides the optimal combination of size freed, network download time, and impact on long-term SSD health.
 
<img width="573" alt="Screen Shot 2022-10-02 at 12 48 13 AM" src="https://user-images.githubusercontent.com/71743241/193438359-27b09d85-85bb-450d-aef2-6ec025eee624.png">

## Day 1

Remove `--info=progress2` in `download_pdf_mmcif.sh`. Instead, just put `--progress`; macOS uses an older `rsync` version that doesn't incrementally recurse files. The output shows the amount of files left, although it floods the console. Downloaded Xcode 14.1 beta 3 because that has `argSort` and `inverse` functions for MPSGraph - functions that AlphaFold uses. I found a very useful hidden utility for debugging MPSGraphs:

```swift
let graph = MPSGraph()
let const = graph.constant(23, dataType: .float16)
graph.perform(NSSelectorFromString("dump")) // prints MLIR
```
```mlir
module  {
  func @main() {
    %0 = "mps.constant"() {value = dense<2.300000e+01> : tensor<f16>} : () -> tensor<f16>
  }
}
```

Here's my game plan:

1) Get the Python code working on CPU-only JAX, comment out multimer code (I'll port that another day).
3) Create a Swift package that calls into Python code, then execute the model in SwiftPM tests.
4) Translate the code outlined below to Swift, verbatim.
5) Replace calls into JAX and TensorFlow with MPSGraph.

| File in `alphafold/model` | Lines to Translate | File in `alphafold/model` | Lines to Translate |
| ------------------------- | ------------------ | ------------------------- | ------------------ |
| all_atom.py               | 1127               | lddt_test.py              | 65                 |
| all_atom_test.py          | 121                | mapping.py                | 209                |
| common_modules.py         | 116                | model.py                  | 163                |
| config.py                 | 643                | modules.py                | 2090               |
| data.py                   | 19                 | prng.py                   | 55                 |
| features.py               | 90                 | prng_test.py              | 32                 |
| folding.py                | 995                | quat_affine.py            | 445                |
| layer_stack.py            | 274                | quat_affine_test.py       | 136                |
| layer_stack_test.py       | 321                | r3.py                     | 306                |
| lddt.py                   | 88                 | utils.py                  | 119                |

| File in `~/geometry`   | Lines to Translate | File in `~/tf`           | Lines to Translate |
| ---------------------- | ------------------ | ------------------------ | ------------------ |
| \_\_init\_\_.py        | 17                 | data_transforms.py       | 611                |
| rigid_matrix_vector.py | 92                 | input_pipeline.py        | 152                |
| rotation_matrix.py     | 143                | protein_features.py      | 115                |
| struct_of_array.py     | 206                | protein_features_test.py | 40                 |
| test_utils.py          | 84                 | proteins_dataset.py      | 152                |
| utils.py               | 9                  | shape_helpers.py         | 33                 |
| vector.py              | 203                | shape_helpers_test.py    | 28                 |
|                        |                    | shape_placeholders.py    | 6                  |
|                        |                    | utils.py                 | 33                 |

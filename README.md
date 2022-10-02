# AlphaFold ported to the M1 GPU

> TODO: Make this overview more concise, emphasize why MLIR is faster than eager

This will be a port of AlphaFold to Swift using Metal Performance Shaders Graph. Although it has been ported to PyTorch (which uses MPSGraph internally), this version can utilize MPSGraph's MLIR compilation capabilities. That may greatly optimize it compared to a purely eager implementation, especially since the model is extremely complex and time-recurrent. Furthermore, writing it in Swift provides full access the low-level Metal API. Metal permits more intense optimizations that aren't possible the more user-friendly PyTorch tensors.

The official AlphaFold repository uses JAX, which graph-compiles code with XLA for optimal performance on Nvidia GPUs. Here, I have optimized it for Apple GPUs so that I can use it on my MacBook Pro for nanotechnology research. These intense optimizations could greatly speedup my workflow - searching for proteins that may be used for transistors or memory cells. These will be proteins that have never been produced before in living organisms, but genetically engineered bacteria can mass-produce them for real-world hardware. Since I don't know what the proteins will be, I need to rapidly search a vast solution space, with minimal latency and no restraints like Colab usage limits.

- Goal: Port the entire framework in a single day using PythonKit and MPSGraph, using https://github.com/lucidrains/alphafold2 as a reference.
- Afterward, compare performance to the Colab notebook that uses an Nvidia K80. Prove it runs faster on my M1 Max, then ridiculously optimize it for inference performance.

## Day 0

Download the reduced dataset, which required 600 GB of disk space. Download `aria2` on Homebrew. Change all instances of `--parents` in the download scripts to `-p`.
 
<img width="573" alt="Screen Shot 2022-10-02 at 12 48 13 AM" src="https://user-images.githubusercontent.com/71743241/193438359-27b09d85-85bb-450d-aef2-6ec025eee624.png">

## Day 1

Remove `--info=progress2` in `download_pdf_mmcif.sh`. Instead, just put `--progress`; macOS uses an older `rsync` version that doesn't incrementally recurse files. The output shows the amount of files left, but it does flood the console. Downloaded Xcode 14.1 beta 3 because that has `argSort` and `inverse` functions for MPSGraph - functions that AlphaFold uses. I found a very useful hidden utility for debugging MPSGraphs:

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

1) Get the Python code working on CPU-only JAX.
2) Create a Swift package that calls into Python code, then execute the model in SwiftPM tests.
3) Translate the code outlined below to Swift, verbatim.
4) Replace calls into JAX and TensorFlow with MPSGraph.

| File in `alphafold/model` | Lines of Python Code |
| ------------------------- | -------------------- |
| all_atom.py               | 1127                 |
| common_modules.py         | 116                  |
| config.py                 | 643                  |
| data.py                   | 19                   |
| features.py               | 90                   |
| folding.py                | 995                  |
| layer_stack.py            | 274                  |
| lddt.py                   | 88                   |
| mapping.py                | 209                  |
| model.py                  | 163                  |
| modules.py                | 2090                 |
| prng.py                   | 55                   |
| quat_affine.py            | 445                  |
| r3.py                     | 306                  |
| utils.py                  | 119                  |

| File in `alphafold/model/geometry` | Lines of Python Code |
| ---------------------------------- | -------------------- |
| \_\_init\_\_.py                    | 17                   |
| rigid_matrix_vector.py             | 92                   |
| rotation_matrix.py                 | 143                  |
| struct_of_array.py                 | 206                  |
| utils.py                           | 9                    |
| vector.py                          | 203                  |

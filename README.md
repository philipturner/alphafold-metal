# AlphaFold ported to the M1 GPU

> TODO: Make this overview more concise.

This will be a port of AlphaFold to Swift using Metal Performance Shaders Graph. Although it has been ported to PyTorch (which uses MPSGraph internally), this version can utilize MPSGraph's MLIR compilation capabilities. That may greatly optimize it compared to a purely eager implementation, especially since the model is time-recurrent and O(n) (based on my current understanding). Furthermore, writing it in Swift provides full access the low-level Metal API. Metal permits more intense optimizations that aren't possible the more user-friendly PyTorch tensors.

The official AlphaFold repository uses JAX, which graph-compiles code with XLA for optimal performance on Nvidia GPUs. Here, I have optimized it for Apple GPUs so that I can use it on my MacBook Pro for nanotechnology research. These intense optimizations could greatly speedup my workflow - searching for proteins that may be used for transistors or memory cells. These will be proteins that have never been produced before in living organisms, but genetically engineered bacteria can mass-produce them for real-world hardware. Since I don't know what the proteins will be, I need to rapidly search a vast solution space, with minimal latency and no restraints like Colab usage limits.

- Goal: Port the entire framework in a single day using PythonKit and MPSGraph, using https://github.com/lucidrains/alphafold2 as a reference.
- Afterward, compare performance to the Colab notebook that uses an Nvidia K80. Prove it runs faster on my M1 Max, then ridiculously optimize it for inference performance.

## Commits by yf225 (Will Feng) by Category

### Benchmarking Infrastructure (27 commits)

Timing measurement methods (profiler mode, cudagraph, etc.) (10 commits):
| Commit | Description |
|--------|-------------|
| 939a884 | Add `--latency-measure-mode=profiler` support ([#386](https://github.com/meta-pytorch/tritonbench/pull/386))|
| 5c633ef | Add `--cudagraph` support to `--latency-measure-mode=profiler` ([#391](https://github.com/meta-pytorch/tritonbench/pull/391))|
| 2c10b64 | Make sure all default-stream produced input tensors have finished writing before starting cudagraph ([#478](https://github.com/meta-pytorch/tritonbench/pull/478))|
| 379a315 | Add L2 cache clearing to do_bench_cudagraph, for more realistic timing ([#519](https://github.com/meta-pytorch/tritonbench/pull/519))|
| 5d05cb9 | Exclude L2 cache clear time from timing measurement ([#527](https://github.com/meta-pytorch/tritonbench/pull/527))|
| 3f896e3 | Profiler mode: use triton do_bench instead of inductor benchmarker for runtime estimation ([#545](https://github.com/meta-pytorch/tritonbench/pull/545))|
| 649df9b | Add ProfilerActivity.CPU to tracked activities ([#401](https://github.com/meta-pytorch/tritonbench/pull/401))|
| 206b93c | Add inductor_benchmarker as latency measurement option ([#333](https://github.com/meta-pytorch/tritonbench/pull/333))|
| f99db62 | Fix missing return_mode arg in _do_bench_inductor ([#390](https://github.com/meta-pytorch/tritonbench/pull/390))|
| c6c6962 | Profiler mode: fix missing CUDA events ([#459](https://github.com/meta-pytorch/tritonbench/pull/459))|

BWD accuracy check improvements (2 commits):
| Commit | Description |
|--------|-------------|
| 40133d8 | Fix BWD gradient check; improved BWD check for rms_norm and layer_norm ([#467](https://github.com/meta-pytorch/tritonbench/pull/467))|
| b80805f | Improve BWD / FWD_BWD accuracy checks; fix layer_norm bwd check ([#414](https://github.com/meta-pytorch/tritonbench/pull/414))|

Better logs / errors (10 commits):
| Commit | Description |
|--------|-------------|
| f686d01 | Print input ID's corresponding input shape after each run ([#643](https://github.com/meta-pytorch/tritonbench/pull/643))|
| cecc55a | Print input ID's corresponding input shape before each run ([#563](https://github.com/meta-pytorch/tritonbench/pull/563))|
| 88fbe13 | Print actual inputs IDs used at warn level ([#562](https://github.com/meta-pytorch/tritonbench/pull/562))|
| daecb64 | Print failing input id when error; print failing backend name ([#510](https://github.com/meta-pytorch/tritonbench/pull/510))|
| 905b152 | Run CUDA synchronize after each _do_bench call, to surface error sooner ([#544](https://github.com/meta-pytorch/tritonbench/pull/544))|
| 3cce791 | Improve failing input error message ([#543](https://github.com/meta-pytorch/tritonbench/pull/543))|
| 4f60f45 | Print exception message when accuracy check fails ([#492](https://github.com/meta-pytorch/tritonbench/pull/492))|
| d24a25f | Fix CSV output bug ([#314](https://github.com/meta-pytorch/tritonbench/pull/314))|
| 95bdbbe | Fix bug in average row display|
| 394bda3 | Fix metric order non-determinism ([#321](https://github.com/meta-pytorch/tritonbench/pull/321))|

More flexibility / control (4 commits):
| Commit | Description |
|--------|-------------|
| 6d8e6e9 | Add --only-match-mode=prefix-with-baseline to enable specific impl prefix together with baseline ([#319](https://github.com/meta-pytorch/tritonbench/pull/319))|
| b8a0ca4 | Add --input-sample-mode CLI flag; allow multiple ids in --input-id ([#476](https://github.com/meta-pytorch/tritonbench/pull/476))|
| eed22b9 | Add benchmark post hook ([#489](https://github.com/meta-pytorch/tritonbench/pull/489))|
| 6938569 | Add --exit-on-exception to exit process on any exception ([#460](https://github.com/meta-pytorch/tritonbench/pull/460))|

Bug fixes (1 commit):
| Commit | Description |
|--------|-------------|
| bde2772 | Avoid calling accelerator sync when CPU only ([#546](https://github.com/meta-pytorch/tritonbench/pull/546))|


### Kernel-Specific Improvements (32 commits)

Attention kernels (6 commits):
| Commit | Description |
|--------|-------------|
| 3b5119a | flash_attention: remove input shape that causes OOM ([#547](https://github.com/meta-pytorch/tritonbench/pull/547))|
| e8647d6 | decoding_attention: gate fbcode import ([#442](https://github.com/meta-pytorch/tritonbench/pull/442))|
| 2555855 | Set baseline for ragged_attention ([#284](https://github.com/meta-pytorch/tritonbench/pull/284))|
| e3f6db6 | Add accuracy check and fixes for fp8_attention Triton kernels ([#276](https://github.com/meta-pytorch/tritonbench/pull/276))|
| 78b71eb | Fix accuracy check for flash_attention kernels ([#280](https://github.com/meta-pytorch/tritonbench/pull/280))|
| d39861a | Add eager and torch.compile impl for gdpa ([#395](https://github.com/meta-pytorch/tritonbench/pull/395))|

Normalization kernels (3 commits):
| Commit | Description |
|--------|-------------|
| 09fe42c | rms_norm: use in_place=False for Liger kernel ([#486](https://github.com/meta-pytorch/tritonbench/pull/486))|
| 70f2688 | rms_norm: pass weight as input args, to reuse the same weight for all impls ([#466](https://github.com/meta-pytorch/tritonbench/pull/466))|
| eeb213c | Fix cudagraph support for rms_norm bwd and layer_norm bwd ([#483](https://github.com/meta-pytorch/tritonbench/pull/483))|

Softmax kernels (2 commits):
| Commit | Description |
|--------|-------------|
| 0938f4d | softmax: change baseline to directly use F.softmax ([#482](https://github.com/meta-pytorch/tritonbench/pull/482))|
| 6216716 | jagged_softmax: Add torch_compile_jagged_softmax_torch_sum for better torch.compile impl ([#481](https://github.com/meta-pytorch/tritonbench/pull/481))|

GEMM kernels (4 commits):
| Commit | Description |
|--------|-------------|
| 2c15edb | gather_gemv: fix eager impl to support large input; fix compile impl ([#455](https://github.com/meta-pytorch/tritonbench/pull/455))|
| 398ea43 | int4_gemm: adjust tolerance, run preproc in measured function, add ([#439](https://github.com/meta-pytorch/tritonbench/pull/439))|
| 9bcfff9 | grouped_gemm: move preparation step into measured function for fair timing comparison ([#431](https://github.com/meta-pytorch/tritonbench/pull/431))|
| a727080 | int4_gemm: improve PyTorch eager impl to match Triton impl behavior ([#399](https://github.com/meta-pytorch/tritonbench/pull/399))|

Cross entropy kernels (3 commits):
| Commit | Description |
|--------|-------------|
| 29404a1 | Pass in weight tensor via kernel function arg for fused_linear_cross_entropy ([#283](https://github.com/meta-pytorch/tritonbench/pull/283))|
| 5a19663 | Allow customizing inputs for cross_entropy benchmark ([#281](https://github.com/meta-pytorch/tritonbench/pull/281))|
| c32f869 | cross_entropy: modify Liger triton kernel to be cudagraph compatible ([#477](https://github.com/meta-pytorch/tritonbench/pull/477))|

Activation kernels (2 commits):
| Commit | Description |
|--------|-------------|
| 487ae9f | Fix geglu accuracy check ([#418](https://github.com/meta-pytorch/tritonbench/pull/418))|
| 2283453 | swiglu: adjust tolerances for bfloat16 dtype ([#422](https://github.com/meta-pytorch/tritonbench/pull/422))|

Reduction kernels (4 commits):
| Commit | Description |
|--------|-------------|
| 7026472 | Add multi-blocks support to sum triton kernel, to enable large input size ([#269](https://github.com/meta-pytorch/tritonbench/pull/269))|
| 812299c | Fix sum kernel's accuracy check method ([#407](https://github.com/meta-pytorch/tritonbench/pull/407))|
| 0067d09 | [welford] Fix torch.compile impl to actually use Welford algorithm ([#592](https://github.com/meta-pytorch/tritonbench/pull/592))|
| 4ec37bd | welford: adjust tolorance to make accuracy check pass ([#285](https://github.com/meta-pytorch/tritonbench/pull/285))|

Other kernels / General changes (8 commits):
| Commit | Description |
|--------|-------------|
| 056ddb0 | Use get_fp8_constants from fp8_utils.py instead of fbgemm_gpu ([#444](https://github.com/meta-pytorch/tritonbench/pull/444))|
| 91cc742 | Use AsyncTaskContext to replace tl.async_task usage ([#288](https://github.com/meta-pytorch/tritonbench/pull/288))|
| 2237731 | Make vector_add Triton version return output tensor, to fix accuracy check|
| a969d61 | Add torch.compile impl to several kernels ([#396](https://github.com/meta-pytorch/tritonbench/pull/396))|
| 87c4375 | Add missing torch.compile impl / improve compile config ([#380](https://github.com/meta-pytorch/tritonbench/pull/380))|
| f034f5c | Rename inductor_ prefix to torch_compile_ ([#398](https://github.com/meta-pytorch/tritonbench/pull/398))|
| 7a99093 | [Benchmark CI] Better kernel prefix filtering, to avoid running unneeded ([#474](https://github.com/meta-pytorch/tritonbench/pull/474))|
| 7c74b21 | Allow custom atol/rtol args for several kernels ([#569](https://github.com/meta-pytorch/tritonbench/pull/569))|
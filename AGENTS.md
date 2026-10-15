# AGENTS.md

## What this repo is
- Header-only C++ fork of MERL's PEAC (AHC fast plane extraction), based on `https://github.com/Urwik/peac`. It is a lab fork with local commits (nanTh, ecDistTh edge filtering, eigenvector/plane-equation fixes), not vanilla upstream.
- Upstream `README.md` is stale: the `cpp/` and `matlab/` folders it describes do not exist. Actual code is `include/` + `src/`.
- Consumed in this workspace by `segmentation_pkgs/arvc_segmentation` (includes it via a hardcoded path `${HOME}/workspaces/arvc_ws/src/third_party/peac`). `COLCON_IGNORE` here and `third_party/CATKIN_IGNORE` mean ROS/catkin never builds this package; it is compiled as headers into the consumer.
- `segmentation_pkgs/depth_plane_seg/include/depth_plane_seg/segmentation/peac/` is a separate vendored copy. Edits here do not affect it.

## Build (standalone demo executables)
- C++17 is mandatory (`src/plane_fitter_pcd.cpp` uses `std::filesystem`) but `CMakeLists.txt` does not set it; a fresh configure without the flag fails with `'filesystem' is not a namespace-name`:
  ```
  cmake -S . -B build -GNinja -DCMAKE_CXX_STANDARD=17
  ninja -C build
  ```
- Targets: `plane_fitter` (live OpenNI/Kinect), `plane_fitter_pcd`, `plane_fitter_pcd_debug`, `pcd_recorder`.
- No tests, CI, linter, or formatter. Verification = build + run on sample clouds in `data/stair` or `data/andreashaus`.

## Running plane_fitter_pcd
- Usage: `./build/plane_fitter_pcd <cloud.pcd> <ini_file>` (see `src/plane_fitter_pcd.tls.ini` for TLS units). With only one arg it loads a hardcoded stale `/home/fran/...` ini, prints "not found", and silently runs with built-in defaults.
- The `list=` key and Windows-style paths in the shipped `.ini` files are ignored by this fork (input comes from argv); `outputDir=` in `src/*.ini` and everything in `config/params.yaml` are stale `/home/fran`/`/home/arvc` paths. The consumer's live config is `arvc_segmentation/config/peac_config.yaml`.
- Units: the fitter assumes mm; `unitScaleFactor` scales input PCD coords (m or cm) to mm.

## Header/API gotchas
- `include/AHCPlaneFitter.hpp` is compiled into consumers, so its top-of-file switches affect them too: `DEBUG`, `DEBUG_INIT`, `DEBUG_CLUSTER`, `DEBUG_AHCLUSTER`, `DEBUG_CALC`, `EVAL_SPEED`, `FILTER_EDGES_BY_LENGTH`. In the current working tree `DEBUG` and `DEBUG_INIT` are enabled (uncommitted); `DEBUG_INIT` draws/writes `db_init*.png` under `PlaneFitter::saveDir`, which the consumer sets from its yaml. Prefer committing intended switch states rather than leaving them dirty.
- Debug images are rendered on a magnified canvas (`PlaneFitter::dbgScale`, computed so each AHC window gets ~12 px, capped at x6; e.g. window=2 on a 512x128 cloud yields 3072x768 PNGs), with fixed thin 1-px marker shapes (dot/ring/x); then they are BGR-converted and saved at that scaled size.
- Fork-only API other packages rely on: `pf.params.nanTh`, `pf.params.ecDistTh` (mm; only honored when `FILTER_EDGES_BY_LENGTH` is defined, currently commented out), `pf.saveDir` (only exists under DEBUG_INIT/DEBUG_CLUSTER), `ahc::OrganizedImage3D` (moved into this header), `MACRO_DEG2RAD`/`MACRO_RAD2DEG`. Grep `arvc_segmentation/src/peac_segmenter_iface.cpp` before renaming/removing any of these.

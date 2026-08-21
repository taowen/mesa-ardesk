# taowen/mesa branch `ardesk`

Mesa `main` plus gallium Freedreno on Qualcomm KGSL. Turnip already
has `tu_knl_kgsl.cc` upstream; `src/freedreno/drm/kgsl/` is this fork.

Build with `-Dfreedreno-kmds=kgsl`. `/dev/kgsl-3d0` is not a DRM node,
so `fd_device_new()` falls back to KGSL when `drmGetVersion` fails.

Does nothing on Mali. Track `upstream/main` (gitlab.freedesktop.org/mesa/mesa)
and keep the kgsl overlay.

import os
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None



class OnnxResidualPolicyWrapper(torch.nn.Module):
    """
    Lightweight ONNX inference wrapper for residual policy: forward(flat_obs) -> residual_action (torch.Tensor)
    Accepts a flat_obs -> obs(np.ndarray) builder through set_obs_builder(builder).
    """

    def __init__(self, onnx_path: str, device: Optional[str] = None):
        super().__init__()
        if ort is None:
            raise ImportError("onnxruntime is not installed, please install onnxruntime first")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Providers: optional TensorRT EP, then CUDA, then CPU
        want_trt = os.environ.get("TENSORRT_EP", "0") not in ("0", "false", "False") or \
                   os.environ.get("ONNX_TRT", "0") not in ("0", "false", "False") or \
                   os.environ.get("ORT_TRT", "0") not in ("0", "false", "False")
        avail = set(getattr(ort, "get_available_providers", lambda: [])())
        providers: list = []
        if torch.cuda.is_available():
            if want_trt and "TensorrtExecutionProvider" in avail:
                trt_opts = {
                    "trt_fp16_enable": os.environ.get("ORT_TRT_FP16", "1") not in ("0", "false", "False"),
                    "trt_int8_enable": os.environ.get("ORT_TRT_INT8", "0") not in ("0", "false", "False"),
                    "trt_engine_cache_enable": os.environ.get("ORT_TRT_CACHE", "1") not in ("0", "false", "False"),
                    "trt_engine_cache_path": os.environ.get("ORT_TRT_CACHE_PATH", ".ort_trt_cache"),
                }
                ws = os.environ.get("ORT_TRT_WORKSPACE", None)
                if ws is not None:
                    try:
                        trt_opts["trt_max_workspace_size"] = int(ws)
                    except Exception:
                        pass
                providers.append(("TensorrtExecutionProvider", trt_opts))
            if "CUDAExecutionProvider" in avail:
                providers.append(("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        providers.append("CPUExecutionProvider")

        debug = os.environ.get("ONNX_DEBUG", "0") not in ("0", "false", "False")
        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        except Exception as e:
            if debug:
                print(f"[ONNX Residual] Session init failed with providers={providers}: {e}. Falling back to CPUExecutionProvider")
            self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        if debug:
            try:
                print(f"[ONNX] Requested providers: {providers}")
                print(f"[ONNX] Session providers:  {self.session.get_providers()}")
                print(f"[ONNX] Device: {self.device}")
            except Exception:
                pass
        self._obs_builder: Optional[Callable[[torch.Tensor], np.ndarray]] = None
        self._input_names: List[str] = [i.name for i in self.session.get_inputs()]
        self._output_names: List[str] = [o.name for o in self.session.get_outputs()]
        self._has_time = ("time_step" in set(self._input_names))
        self.eval()

    def set_obs_builder(self, builder: Callable[[torch.Tensor], np.ndarray]):
        self._obs_builder = builder

    @torch.no_grad()
    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
      
        obs_np=flat_obs
        if isinstance(obs_np, torch.Tensor):
            obs_np = obs_np.detach().cpu().numpy()
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
        inputs = {"obs": obs_np.astype(np.float32)}
        # (1, 183)
        if self._has_time and "time_step" not in inputs:
            inputs["time_step"] = np.zeros((obs_np.shape[0], 1), dtype=np.float32)
        outputs = self.session.run(self._output_names, inputs)
        actions_np = outputs[0]
        return torch.from_numpy(actions_np).to(self.device).float()


class OnnxBasePolicyWrapper(torch.nn.Module):
    """
    ONNX runtime wrapper for FM base policy exported with inputs:
    - real_obs: [B, real_dim]
    - command_obs: [B, cmd_dim]
    - real_historical_obs_raw: [B, hist_dim] or [B, T, real_dim]
    - self_obs: [B, self_dim] (placeholder, values unused by encoder)

    forward(obs_dict) -> actions (torch.Tensor [B, 29])
    If given a non-dict (e.g., warmup tensor), the wrapper will synthesize zero
    inputs of the correct shapes inferred from the ONNX graph.
    """

    def __init__(self, onnx_path: str, device: Optional[str] = None):
        super().__init__()
        if ort is None:
            raise ImportError("onnxruntime is not installed, please install onnxruntime first")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX file does not exist:: {onnx_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Providers: optional TensorRT EP, then CUDA, then CPU
        want_trt = os.environ.get("TENSORRT_EP", "0") not in ("0", "false", "False") or \
                   os.environ.get("ONNX_TRT", "0") not in ("0", "false", "False") or \
                   os.environ.get("ORT_TRT", "0") not in ("0", "false", "False")
        avail = set(getattr(ort, "get_available_providers", lambda: [])())
        providers: list = []
        if torch.cuda.is_available():
            if want_trt and "TensorrtExecutionProvider" in avail:
                trt_opts = {
                    "trt_fp16_enable": os.environ.get("ORT_TRT_FP16", "1") not in ("0", "false", "False"),
                    "trt_int8_enable": os.environ.get("ORT_TRT_INT8", "0") not in ("0", "false", "False"),
                    "trt_engine_cache_enable": os.environ.get("ORT_TRT_CACHE", "1") not in ("0", "false", "False"),
                    "trt_engine_cache_path": os.environ.get("ORT_TRT_CACHE_PATH", ".ort_trt_cache"),
                }
                ws = os.environ.get("ORT_TRT_WORKSPACE", None)
                if ws is not None:
                    try:
                        trt_opts["trt_max_workspace_size"] = int(ws)
                    except Exception:
                        pass
                providers.append(("TensorrtExecutionProvider", trt_opts))
            if "CUDAExecutionProvider" in avail:
                providers.append(("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        providers.append("CPUExecutionProvider")
        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        # cache IO names and dims
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._name_to_dim = {}
        for i in self.session.get_inputs():
            # i.shape is like [None, 90]
            dims = []
            try:
                for d in i.shape:
                    if isinstance(d, int):
                        dims.append(d)
                    else:
                        # dynamic dim or str -> None
                        dims.append(None)
            except Exception:
                dims = [None, None]
            self._name_to_dim[i.name] = dims
        self._obs_builder: Optional[Callable[[torch.Tensor], dict]] = None
        self.eval()

    def set_obs_builder(self, builder: Callable[[torch.Tensor], dict]):
        self._obs_builder = builder

    def _infer_feat_dim(self, name: str, fallback: int = None) -> Optional[int]:
        dims = self._name_to_dim.get(name)
        if dims and len(dims) >= 2:
            return dims[1]
        return fallback

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _can_iobind(self) -> bool:
        try:
            prov = set(self.session.get_providers())
        except Exception:
            prov = set()
        has_dlpack_ortvalue = hasattr(ort, "OrtValue") and hasattr(ort.OrtValue, "from_dlpack")
        return (
            self.device.type == "cuda"
            and ("CUDAExecutionProvider" in prov or "TensorrtExecutionProvider" in prov)
            and has_dlpack_ortvalue
        )

    def _action_dim(self) -> int:
        try:
            shp = self.session.get_outputs()[0].shape
            if len(shp) >= 2 and isinstance(shp[1], int) and shp[1] is not None:
                return int(shp[1])
        except Exception:
            pass
        return 29

    @torch.no_grad()
    def forward(self, obs_or_flat: Any) -> torch.Tensor:
        # Build obs dict either from provided dict or via builder
        obs_dict: Optional[dict] = None
        if isinstance(obs_or_flat, dict):
            obs_dict = obs_or_flat
        elif self._obs_builder is not None:
            obs_dict = self._obs_builder(obs_or_flat)

        inputs: dict = {}
        if obs_dict is not None:
            # Gather tensors, allow history as [B, T, D] or [B, T*D]
            real_obs = obs_dict.get("real_obs")
            command_obs = obs_dict.get("command_obs")
            real_hist = obs_dict.get("real_historical_obs_raw")
            self_obs = obs_dict.get("self_obs", None)

            def to_np(x: Any) -> np.ndarray:
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy().astype(np.float32)
                return np.asarray(x, dtype=np.float32)

            if real_obs is None or command_obs is None or real_hist is None:
                raise ValueError("obs_dict 缺少必要键：real_obs/command_obs/real_historical_obs_raw")

            # Prefer GPU tensors if possible for zero-copy
            def _build_cpu_inputs():
                ro = self._ensure_2d(to_np(real_obs))
                co = self._ensure_2d(to_np(command_obs))
                rh = to_np(real_hist)
                if rh.ndim == 3:
                    B_, T_, D_ = rh.shape
                    rh = rh.reshape(B_, T_ * D_)
                rh = self._ensure_2d(rh)

                B_ = ro.shape[0]
                if self_obs is None:
                    self_dim = self._infer_feat_dim("self_obs", 1) or 1
                    so = np.zeros((B_, int(self_dim)), dtype=np.float32)
                else:
                    so = self._ensure_2d(to_np(self_obs))

                inputs_local: Dict[str, np.ndarray] = {
                    "real_obs": ro.astype(np.float32),
                    "command_obs": co.astype(np.float32),
                    "real_historical_obs_raw": rh.astype(np.float32),
                    "self_obs": so.astype(np.float32),
                }
                # Optional external noise input for TRT-friendly export
                if "initial_noise" in self._input_names:
                    init = obs_dict.get("initial_noise", None)
                    if init is None:
                        d_init = self._infer_feat_dim("initial_noise", self._action_dim()) or self._action_dim()
                        init_np = np.zeros((B_, int(d_init)), dtype=np.float32)
                    else:
                        init_np = to_np(init)
                        if init_np.ndim == 1:
                            init_np = init_np[None, :]
                        elif init_np.ndim > 2:
                            # Flatten trailing dims if [B,1,D] etc.
                            init_np = init_np.reshape(init_np.shape[0], -1)
                    inputs_local["initial_noise"] = init_np.astype(np.float32)

                return inputs_local

            if self._can_iobind():
                # Ensure tensors on GPU and float32, contiguous
                def to_gpu_f32(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    t = t.to(self.device, dtype=torch.float32)
                    return t.contiguous()
                try:
                    ro_t = to_gpu_f32(real_obs)
                    co_t = to_gpu_f32(command_obs)
                    rh_t = real_hist
                    if isinstance(rh_t, torch.Tensor) and rh_t.ndim == 3:
                        rh_t = rh_t.reshape(rh_t.shape[0], -1)
                    rh_t = to_gpu_f32(rh_t)
                    B = int(ro_t.shape[0])
                    if self_obs is None:
                        self_dim = self._infer_feat_dim("self_obs", 1) or 1
                        so_t = torch.zeros((B, int(self_dim)), device=self.device, dtype=torch.float32)
                    else:
                        so_t = to_gpu_f32(self_obs)

                    # Build IO binding
                    io = self.session.io_binding()
                    from torch.utils.dlpack import to_dlpack
                    def bind_input(name: str, t: torch.Tensor):
                        ov = ort.OrtValue.from_dlpack(to_dlpack(t))
                        io.bind_ortvalue_input(name, ov)

                    bind_input("real_obs", ro_t)
                    bind_input("command_obs", co_t)
                    bind_input("real_historical_obs_raw", rh_t)
                    bind_input("self_obs", so_t)

                    # Optional external noise input
                    if "initial_noise" in self._input_names:
                        if "initial_noise" in obs_dict:
                            init_t = to_gpu_f32(obs_dict["initial_noise"])
                            if init_t.ndim > 2:
                                init_t = init_t.view(init_t.shape[0], -1)
                        else:
                            d_init = int(self._infer_feat_dim("initial_noise", self._action_dim()) or self._action_dim())
                            init_t = torch.zeros((B, d_init), device=self.device, dtype=torch.float32)
                        bind_input("initial_noise", init_t)

                    # Prepare output buffer on GPU
                    act_dim = self._action_dim()
                    out_t = torch.empty((B, act_dim), device=self.device, dtype=torch.float32)
                    io.bind_ortvalue_output(self._output_names[0], ort.OrtValue.from_dlpack(to_dlpack(out_t)))
                    if os.environ.get("ONNX_DEBUG", "0") not in ("0", "false", "False"):
                        print("[ONNX] Using IO-binding (GPU zero-copy) for base policy")
                    self.session.run_with_iobinding(io)
                    return out_t
                except Exception:
                    # Fallback to numpy path if IO-binding is not supported
                    if os.environ.get("ONNX_DEBUG", "0") not in ("0", "false", "False"):
                        print("[ONNX] IO-binding failed, falling back to CPU numpy path")
                    inputs = _build_cpu_inputs()
            else:
                # CPU fallback with numpy
                inputs = _build_cpu_inputs()
        else:
            # Warmup or unknown input: synthesize zeros based on ONNX input dims
            # Determine batch size from tensor if available
            B = 1
            if isinstance(obs_or_flat, torch.Tensor) and obs_or_flat.ndim >= 1:
                B = int(obs_or_flat.shape[0])
            if self._can_iobind():
                from torch.utils.dlpack import to_dlpack
                def mk_t(name: str) -> torch.Tensor:
                    d = int(self._infer_feat_dim(name, 1) or 1)
                    return torch.zeros((B, d), device=self.device, dtype=torch.float32)
                ro_t = mk_t("real_obs")
                co_t = mk_t("command_obs")
                rh_t = mk_t("real_historical_obs_raw")
                so_t = mk_t("self_obs")
                io = self.session.io_binding()
                io.bind_ortvalue_input("real_obs", ort.OrtValue.from_dlpack(to_dlpack(ro_t)))
                io.bind_ortvalue_input("command_obs", ort.OrtValue.from_dlpack(to_dlpack(co_t)))
                io.bind_ortvalue_input("real_historical_obs_raw", ort.OrtValue.from_dlpack(to_dlpack(rh_t)))
                io.bind_ortvalue_input("self_obs", ort.OrtValue.from_dlpack(to_dlpack(so_t)))
                if "initial_noise" in self._input_names:
                    init_t = mk_t("initial_noise")
                    io.bind_ortvalue_input("initial_noise", ort.OrtValue.from_dlpack(to_dlpack(init_t)))
                act_dim = self._action_dim()
                out_t = torch.empty((B, act_dim), device=self.device, dtype=torch.float32)
                io.bind_ortvalue_output(self._output_names[0], ort.OrtValue.from_dlpack(to_dlpack(out_t)))
                self.session.run_with_iobinding(io)
                return out_t
            else:
                def mk(name: str) -> np.ndarray:
                    d = self._infer_feat_dim(name, 1) or 1
                    return np.zeros((B, int(d)), dtype=np.float32)
                inputs = {
                    "real_obs": mk("real_obs"),
                    "command_obs": mk("command_obs"),
                    "real_historical_obs_raw": mk("real_historical_obs_raw"),
                    "self_obs": mk("self_obs"),
                }
                if "initial_noise" in self._input_names:
                    inputs["initial_noise"] = mk("initial_noise")

        # Filter inputs to only those actually present in the ONNX graph
        if isinstance(inputs, dict):
            valid = set(self._input_names)
            inputs = {k: v for k, v in inputs.items() if k in valid}

        outputs = self.session.run(self._output_names, inputs)
        actions_np = outputs[0]
        return torch.from_numpy(actions_np).to(self.device).float()

    def to(self, device: Any):  # noop-style to() to match nn.Module ergonomics
        self.device = torch.device(device)
        return self

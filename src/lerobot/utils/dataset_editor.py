#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Tkinter-based GUI editor for LeRobot datasets.

Provides three main features:
    1. **Preview** a dataset: scrub through episodes, view camera frames, and plot
       the action / observation.state joint trajectories over time.
    2. **Edit episodes**: mark individual episodes for deletion and write the result
       to a new dataset (non-destructive by default).
    3. **Combine datasets**: merge several datasets (given by Hugging Face URL,
       ``repo_id``, or local path) into a single new dataset.

The GUI reuses the same building blocks as the ``lerobot-edit-dataset`` CLI
(:func:`lerobot.datasets.delete_episodes` and
:func:`lerobot.datasets.merge_datasets`), so behaviour matches the CLI exactly.

Run with::

    lerobot-dataset-editor
    python -m lerobot.utils.dataset_editor

Requires an interactive display and the ``dataset`` extra (``av``, ``datasets``).
Matplotlib and Tkinter are used for the UI (no heavy Qt dependency).
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import threading
import tkinter as tk
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_HF_HOSTS = ("huggingface.co", "hf.co", "www.huggingface.co", "www.hf.co")
ACTION_KEY = "action"
STATE_KEY = "observation.state"


# ──────────────────────────────────────────────────────────────────────────────
# Backend helpers (GUI-free, reusable)
# ──────────────────────────────────────────────────────────────────────────────


def normalize_hub_source(src: str) -> tuple[str, Path | None]:
    """Normalize a user-entered dataset reference into ``(repo_id, root)``.

    Accepts:
        - A Hugging Face URL: ``https://huggingface.co/datasets/user/ds`` (also
          ``hf.co``, ``/tree/main``, ``/blob/...``, trailing ``.git``).
        - A bare ``repo_id``: ``user/ds``.
        - A local directory containing a ``meta/info.json`` (used as ``root``).

    Returns:
        ``(repo_id, root)`` where ``root`` is a ``Path`` for local datasets or
        ``None`` to let :class:`LeRobotDataset` resolve/download it.
    """
    src = src.strip()
    if not src:
        raise ValueError("Empty dataset reference.")

    # Local directory?
    candidate = Path(src)
    if (candidate / "meta" / "info.json").exists():
        return candidate.name, candidate.resolve()

    # URL? Only when it has a scheme or clearly references the HF hub.
    is_url = "://" in src or src.lower().startswith(_HF_HOSTS) or "/datasets/" in src
    if is_url:
        parsed = urlparse(src if "://" in src else "https://" + src)
        path = parsed.path or ""
        if "/datasets/" in path:
            path = re.sub(r"/datasets/", "/", path, count=1)
        path = re.sub(r"\.git$", "", path)
        path = re.sub(r"/(?:tree|blob|resolve|raw)/.*$", "", path)
        repo_id = path.strip("/").split("?")[0]
        return repo_id, None

    # Bare repo_id (e.g. ``user/ds``) or local name.
    repo_id = re.sub(r"\.git$", "", src).strip("/")
    return repo_id, None


def _resolve_output(
    default_repo_id: str, new_repo_id: str | None, new_root: str | None
) -> tuple[str, Path]:
    """Resolve output repo_id and directory with the same defaults as the CLI."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    repo_id = new_repo_id or default_repo_id
    output_dir = Path(new_root) if new_root else HF_LEROBOT_HOME / repo_id
    return repo_id, output_dir.resolve()


def load_dataset(repo_id: str, root: Path | None = None):
    """Lazily import and construct a :class:`LeRobotDataset`."""
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset(repo_id, root=root)


def load_dataset_meta(repo_id: str, root: Path | None = None):
    """Load only the lightweight metadata (no parquet data), for quick counts."""
    from lerobot.datasets import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id, root=root)


def episode_rows(dataset) -> list[dict]:
    """Return one dict per episode: ``{index, length, task}`` for display."""
    meta = dataset.meta
    episodes = meta.episodes
    if episodes is None:
        return []
    rows = []
    for i in range(len(episodes)):
        ep = episodes[i]
        tasks_field = ep.get("tasks", [])
        if isinstance(tasks_field, (list, tuple)) and tasks_field:
            task = ", ".join(str(t) for t in tasks_field)
        else:
            task = ""
        rows.append(
            {
                "index": int(ep["episode_index"]),
                "length": int(ep["length"]),
                "task": task,
            }
        )
    return rows


def get_feature_names(dataset, key: str) -> list[str]:
    """Return per-dimension names for a vector feature.

    Handles flat-list ``names`` and dict-style ``names`` (e.g.
    ``{"joints": ["j0", "j1"]}``); falls back to ``{key}_{i}`` otherwise.
    """
    feature = dataset.features[key]
    dim = feature["shape"][-1]
    names = feature.get("names")
    flat: list[str] = []
    if isinstance(names, list) and len(names) == dim:
        flat = [str(n) for n in names]
    elif isinstance(names, dict):
        for vals in names.values():
            if isinstance(vals, (list, tuple)):
                flat.extend(str(v) for v in vals)
        if len(flat) != dim:
            flat = []
    if len(flat) != dim:
        return [f"{key}_{d}" for d in range(dim)]
    return flat


def tensor_to_hwc_uint8(chw_tensor):
    """Convert a CHW float32 tensor in [0,1] to an HWC uint8 numpy array."""
    import torch

    t = chw_tensor if isinstance(chw_tensor, torch.Tensor) else torch.as_tensor(chw_tensor)
    t = t.detach().cpu().float()
    if t.ndim == 3 and t.shape[0] <= 4:  # CHW
        t = t.permute(1, 2, 0)
    return (t * 255).clamp(0, 255).to(torch.uint8).numpy()


def frame_to_pil(dataset, key: str, chw_tensor):
    """Convert a decoded frame tensor to a PIL image, applying a colormap to depth."""
    import numpy as np
    from PIL import Image

    if key in getattr(dataset.meta, "depth_keys", []):
        arr = chw_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
        depth = arr[:, :, 0]
        valid = depth[depth > 0]
        lo = float(valid.min()) if valid.size else 0.0
        hi = float(valid.max()) if valid.size else 1.0
        norm = np.nan_to_num((depth - lo) / (hi - lo + 1e-9))
        try:
            from matplotlib import colormaps

            rgb = (colormaps["viridis"](norm)[..., :3] * 255).astype(np.uint8)
        except Exception:  # noqa: BLE001
            rgb = (np.clip(norm, 0, 1)[..., None].repeat(3, axis=-1) * 255).astype(np.uint8)
        return Image.fromarray(rgb)
    return Image.fromarray(tensor_to_hwc_uint8(chw_tensor))


def run_delete_episodes(
    dataset,
    episode_indices: list[int],
    new_repo_id: str | None,
    new_root: str | None,
):
    """Wrap :func:`delete_episodes` with in-place backup logic (matches CLI).

    When the source dataset's videos are encoded with SVT-AV1, the re-encode that
    ``delete_episodes`` performs on shared video files crashes the bundled FFmpeg
    build during teardown. To avoid that, we transparently re-encode affected
    videos with libx264 (h264) instead — universally decodable and crash-free.
    """
    from lerobot.configs import (
        depth_encoder_defaults,
        rgb_encoder_defaults,
    )
    from lerobot.datasets import delete_episodes

    repo_id, output_dir = _resolve_output(
        default_repo_id=f"{dataset.repo_id}_edited",
        new_repo_id=new_repo_id,
        new_root=new_root,
    )
    input_path = Path(dataset.root).resolve()
    in_place = output_dir == input_path
    if in_place:
        backup = input_path.with_name(input_path.name + "_old")
        logger.warning("In-place edit: backing up %s -> %s", input_path, backup)
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(input_path), str(backup))
        dataset.root = input_path.with_name(input_path.name + "_old")

    # Detect SVT-AV1 source videos and swap the re-encode codec to h264.
    # The native SVT-AV1 encoder segfaults during teardown when multiple
    # re-encode lifecycles run in one process (per the editor's delete flow).
    rgb_enc = None
    depth_enc = None
    svt_keys = []
    for key in getattr(dataset.meta, "video_keys", []):
        info = dataset.meta.features.get(key, {}).get("info", {})
        codec = info.get("video.codec") or info.get("codec")
        if codec in ("libsvtav1", "av1", "svt-av1"):
            svt_keys.append(key)
    if svt_keys:
        logger.warning(
            "Source videos %s are SVT-AV1, which crashes during re-encode on this "
            "FFmpeg build. Re-encoding deleted-episode videos as h264 instead.",
            svt_keys,
        )
        # Build fresh configs and set per-codec fields explicitly: the SVT-AV1
        # defaults (numeric preset=12, hevc/x265 extra_options) are invalid for
        # libx264 and would make avcodec_open2 fail.
        rgb_enc = rgb_encoder_defaults()
        rgb_enc.vcodec = "libx264"
        rgb_enc.pix_fmt = "yuv420p"
        rgb_enc.crf = 23
        rgb_enc.preset = "medium"  # libx264 uses string presets
        rgb_enc.extra_options = {}
        depth_enc = depth_encoder_defaults()
        depth_enc.vcodec = "libx264"
        depth_enc.pix_fmt = "yuv420p"
        depth_enc.crf = 18  # higher quality for depth maps (preserve precision)
        depth_enc.preset = "medium"
        depth_enc.extra_options = {}  # drop the x265 lossless params

    return delete_episodes(
        dataset,
        episode_indices=sorted(episode_indices),
        output_dir=output_dir,
        repo_id=repo_id,
        rgb_encoder=rgb_enc,
        depth_encoder=depth_enc,
    )


def run_merge(
    sources: list[tuple[str, Path | None]],
    new_repo_id: str,
    new_root: str | None,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
):
    """Load each source dataset and merge them into one new dataset."""
    from lerobot.datasets import LeRobotDataset, merge_datasets

    if not sources:
        raise ValueError("No source datasets to merge.")
    if not new_repo_id:
        raise ValueError("Output repo_id is required for merge.")

    repo_id, output_dir = _resolve_output(
        default_repo_id=new_repo_id, new_repo_id=new_repo_id, new_root=new_root
    )
    datasets = [LeRobotDataset(rid, root=root) for (rid, root) in sources]
    return merge_datasets(
        datasets,
        output_repo_id=repo_id,
        output_dir=output_dir,
        concatenate_videos=concatenate_videos,
        concatenate_data=concatenate_data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────


class _QueueLogHandler(logging.Handler):
    """Thread-safe logging handler: puts formatted records on a queue."""

    def __init__(self, q: queue.Queue[str]):
        super().__init__()
        self.queue = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


class DatasetEditorApp:
    """Main dataset-editor window (built on top of a ``tk.Tk`` root)."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.dataset = None  # currently loaded LeRobotDataset (preview/edit target)
        self._busy_lock = threading.Lock()
        self._busy = False
        self._play_after_id: str | None = None
        self._play_playing = False
        self._play_cursor = -1  # last shown frame during playback
        self._slider_programmatic = False  # suppresses _on_slider during playback
        # Prefetch pipeline for smooth playback (populated only while playing).
        self._frame_queue: queue.Queue[tuple[int, dict[str, object]]] = queue.Queue(maxsize=8)
        self._producer_thread: threading.Thread | None = None
        self._producer_gen = 0  # bumped to cancel an in-flight producer

        root.title("LeRobot Dataset Editor")
        root.geometry("1280x820")
        root.minsize(960, 640)

        self._build_top_bar()
        self._build_notebook()
        self._build_status_bar()

        # Thread-safe log capture -> status text widget.
        self.log_queue: queue.Queue[str] = queue.Queue()
        handler = _QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self._log_handler = handler
        logging.getLogger("lerobot").addHandler(handler)

        # Thread-safe async-result dispatch: worker threads post results here and
        # the main thread polls (Tkinter's `after` is not safe to call off-thread).
        self._result_queue: queue.Queue[
            tuple[Callable[[object, BaseException | None], None], object, BaseException | None]
        ] = queue.Queue()

        root.after(100, self._drain_logs)
        root.after(100, self._drain_results)

    # ── Top bar ────────────────────────────────────────────────────────────

    def _build_top_bar(self) -> None:
        bar = ttk.LabelFrame(self.root, text="Dataset")
        bar.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(bar, text="Source:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
        self.src_var = tk.StringVar()
        self.src_entry = ttk.Entry(bar, textvariable=self.src_var, width=52)
        self.src_entry.grid(row=0, column=1, sticky="we", padx=4)
        self.src_entry.focus_set()

        self.load_btn = ttk.Button(bar, text="Load", command=self.on_load)
        self.load_btn.grid(row=0, column=2, padx=4)
        self.src_entry.bind("<Return>", lambda _e: self.on_load())

        self.summary_var = tk.StringVar(value="No dataset loaded.")
        ttk.Label(bar, textvariable=self.summary_var, anchor="w").grid(
            row=1, column=0, columnspan=3, sticky="we", padx=8, pady=(0, 6)
        )

        out = ttk.LabelFrame(self.root, text="Output (for Edit / Combine)")
        out.pack(fill="x", padx=8, pady=4)
        ttk.Label(out, text="repo_id:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
        self.out_repo_var = tk.StringVar()
        ttk.Entry(out, textvariable=self.out_repo_var, width=36).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(out, text="directory:").grid(row=0, column=2, sticky="w", padx=(8, 4))
        self.out_root_var = tk.StringVar()
        ttk.Entry(out, textvariable=self.out_root_var, width=36).grid(row=0, column=3, sticky="we", padx=4)
        ttk.Button(out, text="Browse…", command=self._browse_out_root).grid(row=0, column=4, padx=4)
        self.push_btn = ttk.Button(out, text="Push to Hub…", command=self.on_push_to_hub)
        self.push_btn.grid(row=0, column=5, padx=4)
        out.columnconfigure(1, weight=1)
        out.columnconfigure(3, weight=1)
        bar.columnconfigure(1, weight=1)

    def _browse_out_root(self) -> None:
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.out_root_var.set(d)

    # ── Notebook / tabs ────────────────────────────────────────────────────

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self._build_preview_tab()
        self._build_edit_tab()
        self._build_combine_tab()

    def _build_preview_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Preview")

        # Left: episode list.
        left = ttk.Frame(tab, width=300)
        left.pack(side="left", fill="y", padx=(4, 2), pady=4)
        left.pack_propagate(False)
        ttk.Label(left, text="Episodes").pack(anchor="w", padx=4, pady=(4, 2))
        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(side="left", fill="y", expand=True)
        self.prev_tree = ttk.Treeview(
            tree_wrap, columns=("frames", "task"), show="tree headings", height=24
        )
        self.prev_tree.heading("#0", text="idx")
        self.prev_tree.heading("frames", text="frames")
        self.prev_tree.heading("task", text="task")
        self.prev_tree.column("#0", width=50, stretch=False)
        self.prev_tree.column("frames", width=60, stretch=False)
        self.prev_tree.column("task", width=180)
        self.prev_tree.pack(side="left", fill="y", expand=True)
        sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.prev_tree.yview)
        self.prev_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.prev_tree.bind("<<TreeviewSelect>>", lambda _e: self.on_preview_select_episode())

        # Right: image + plot.
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=(2, 4), pady=4)

        self.cam_frame = ttk.Frame(right)
        self.cam_frame.pack(fill="both", expand=True)
        self.cam_frame_inner: ttk.Frame | None = None
        self._camera_labels: dict[str, ttk.Label] = {}
        self._placeholder_label = ttk.Label(self.cam_frame, text="Load a dataset and select an episode.")
        self._placeholder_label.pack(expand=True)

        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(4, 2))
        self.frame_slider = ttk.Scale(
            controls, from_=0, to=1, orient="horizontal", command=self._on_slider
        )
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.frame_label_var = tk.StringVar(value="frame -")
        ttk.Label(controls, textvariable=self.frame_label_var, width=18).pack(side="left", padx=4)
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_play, state="disabled")
        self.play_btn.pack(side="left", padx=4)

        # Matplotlib figure (action + state).
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(7, 3.4), dpi=100)
        self.ax_action = self.fig.add_subplot(211)
        self.ax_state = self.fig.add_subplot(212, sharex=self.ax_action)
        self.fig.subplots_adjust(hspace=0.35, left=0.10, right=0.97, top=0.95, bottom=0.12)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl_canvas.get_tk_widget().pack(fill="x", padx=4, pady=(4, 0))
        self._vline_action = None
        self._vline_state = None
        self._blit_bg = None
        self._blit_bbox = None

        # Per-episode cached state.
        self._ep_start: int | None = None
        self._ep_end: int | None = None
        self._ep_len: int = 0
        self._ep_fps: int = 1

    def _build_edit_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Edit episodes")

        tree_wrap = ttk.Frame(tab)
        tree_wrap.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.edit_tree = ttk.Treeview(
            tree_wrap, columns=("frames", "task"), show="tree headings", selectmode="none"
        )
        self.edit_tree.heading("#0", text="del")
        self.edit_tree.heading("frames", text="frames")
        self.edit_tree.heading("task", text="task")
        self.edit_tree.column("#0", width=50, stretch=False, anchor="center")
        self.edit_tree.column("frames", width=70, stretch=False, anchor="e")
        self.edit_tree.column("task", width=600)
        self.edit_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.edit_tree.yview)
        self.edit_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        # Click on the first column (#0) toggles the mark.
        self.edit_tree.bind("<Button-1>", self._on_edit_click)
        self._edit_marks: set[int] = set()

        bottom = ttk.Frame(tab)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.edit_status_var = tk.StringVar(value="No dataset loaded.")
        ttk.Label(bottom, textvariable=self.edit_status_var).pack(side="left")
        self.delete_btn = ttk.Button(
            bottom, text="Delete marked episodes", command=self.on_delete_marked, state="disabled"
        )
        self.delete_btn.pack(side="right")

    def _build_combine_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Combine")

        top = ttk.Frame(tab)
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(top, text="Add source:").pack(side="left")
        self.merge_src_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.merge_src_var, width=50)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        self.add_src_btn = ttk.Button(top, text="Add", command=self.on_add_source)
        self.add_src_btn.pack(side="left", padx=(4, 0))
        entry.bind("<Return>", lambda _e: self.on_add_source())

        tree_wrap = ttk.Frame(tab)
        tree_wrap.pack(fill="both", expand=True, padx=8, pady=4)
        self.merge_tree = ttk.Treeview(
            tree_wrap, columns=("episodes", "frames", "root"), show="tree headings", height=12
        )
        self.merge_tree.heading("#0", text="source (repo_id)")
        self.merge_tree.heading("episodes", text="episodes")
        self.merge_tree.heading("frames", text="frames")
        self.merge_tree.heading("root", text="root")
        self.merge_tree.column("#0", width=280)
        self.merge_tree.column("episodes", width=90, anchor="e")
        self.merge_tree.column("frames", width=90, anchor="e")
        self.merge_tree.column("root", width=320)
        self.merge_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.merge_tree.yview)
        self.merge_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._merge_sources: list[tuple[str, Path | None]] = []

        mid = ttk.Frame(tab)
        mid.pack(fill="x", padx=8, pady=2)
        ttk.Button(mid, text="Remove selected", command=self.on_remove_source).pack(side="left")
        self.concat_videos_var = tk.BooleanVar(value=True)
        self.concat_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mid, text="concatenate videos", variable=self.concat_videos_var).pack(
            side="left", padx=16
        )
        ttk.Checkbutton(mid, text="concatenate data", variable=self.concat_data_var).pack(side="left")

        bottom = ttk.Frame(tab)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.merge_status_var = tk.StringVar(value="No sources added.")
        ttk.Label(bottom, textvariable=self.merge_status_var).pack(side="left")
        self.merge_btn = ttk.Button(
            bottom, text="Merge into new dataset", command=self.on_merge, state="disabled"
        )
        self.merge_btn.pack(side="right")

    def _build_status_bar(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Log")
        frame.pack(fill="x", padx=8, pady=(4, 8))
        inner = ttk.Frame(frame)
        inner.pack(fill="both", expand=True, padx=(4, 4), pady=4)
        self.log_text = tk.Text(inner, height=6, state="disabled", wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    # ── Busy state ─────────────────────────────────────────────────────────

    def set_busy(self, busy: bool) -> None:
        with self._busy_lock:
            self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in (
            self.load_btn,
            self.delete_btn,
            self.add_src_btn,
            self.merge_btn,
            self.play_btn,
            self.push_btn,
        ):
            with suppress(Exception):
                btn.configure(state=state)

    def run_async(
        self,
        work: Callable[[], object],
        on_done: Callable[[object, BaseException | None], None],
    ) -> None:
        """Run ``work`` in a worker thread; call ``on_done(result, error)`` on the UI thread.

        The worker never touches Tk directly — it posts the result onto
        ``_result_queue`` and ``_drain_results`` (polled on the main thread)
        invokes ``on_done``. This is the only thread-safe way to cross from a
        Python thread back into Tkinter.
        """

        def _runner() -> None:
            err: BaseException | None = None
            res: object = None
            try:
                res = work()
            except BaseException as e:  # noqa: BLE001
                err = e
                logger.exception("Operation failed")
            self._result_queue.put_nowait((on_done, res, err))

        self.set_busy(True)
        threading.Thread(target=_runner, daemon=True).start()

    def _drain_results(self) -> None:
        try:
            while True:
                on_done, res, err = self._result_queue.get_nowait()
                on_done(res, err)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_results)

    def _drain_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_logs)

    # ── Load ───────────────────────────────────────────────────────────────

    def on_load(self) -> None:
        src = self.src_var.get().strip()
        if not src:
            messagebox.showwarning("Load dataset", "Enter a repo_id, HF URL, or local path.")
            return
        try:
            repo_id, root = normalize_hub_source(src)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load dataset", f"Invalid reference:\n{e}")
            return

        logger.info("Loading dataset %s (root=%s)", repo_id, root)

        def work():
            return load_dataset(repo_id, root)

        def done(ds, err):
            self.set_busy(False)
            if err is not None:
                messagebox.showerror("Load dataset", f"Failed to load:\n{err}")
                return
            self._on_dataset_loaded(ds)

        self.run_async(work, done)

    def _on_dataset_loaded(self, ds) -> None:
        self.dataset = ds
        rows = episode_rows(ds)
        self.summary_var.set(
            f"{ds.repo_id}  |  {ds.meta.total_episodes} episodes  |  "
            f"{ds.meta.total_frames} frames  |  {ds.meta.fps} fps  |  root: {ds.root}"
        )
        # Default output = new dataset to avoid clobbering.
        if not self.out_repo_var.get():
            self.out_repo_var.set(f"{ds.repo_id.replace('/', '_')}_edited")
        if not self.out_root_var.get():
            from lerobot.utils.constants import HF_LEROBOT_HOME

            self.out_root_var.set(str(HF_LEROBOT_HOME / self.out_repo_var.get()))

        # Populate preview + edit lists.
        self._populate_preview_tree(rows)
        self._populate_edit_tree(rows)
        logger.info(
            "Loaded %s: %s episodes, %s frames", ds.repo_id, ds.meta.total_episodes, ds.meta.total_frames
        )

    def _populate_preview_tree(self, rows: list[dict]) -> None:
        self.prev_tree.delete(*self.prev_tree.get_children())
        for r in rows:
            self.prev_tree.insert(
                "", "end", iid=str(r["index"]), text=str(r["index"]), values=(r["length"], r["task"])
            )

    def _populate_edit_tree(self, rows: list[dict]) -> None:
        self.edit_tree.delete(*self.edit_tree.get_children())
        self._edit_marks.clear()
        for r in rows:
            self.edit_tree.insert(
                "", "end", iid=str(r["index"]), text="☐", values=(r["length"], r["task"]), tags=("row",)
            )
        self.delete_btn.configure(state="normal" if rows else "disabled")
        self._refresh_edit_status(rows)

    def _episodes_rows(self) -> list[dict]:
        if self.dataset is None:
            return []
        return episode_rows(self.dataset)

    # ── Preview tab ────────────────────────────────────────────────────────

    def on_preview_select_episode(self) -> None:
        if self.dataset is None:
            return
        sel = self.prev_tree.selection()
        if not sel:
            return
        ep_idx = int(sel[0])
        self._stop_play()

        def done(res, err):
            self.set_busy(False)
            if err is not None:
                messagebox.showerror("Preview", f"Failed to load episode {ep_idx}:\n{err}")
                return
            start, end, action_arr, state_arr = res
            self._apply_episode(start, end, action_arr, state_arr)

        self.run_async(lambda: self._load_episode(ep_idx), done)

    def _load_episode(self, ep_idx: int):
        """Worker: read episode bounds + action/state arrays (no video decode)."""
        import numpy as np

        ds = self.dataset
        ep = ds.meta.episodes[ep_idx]
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        cols = [c for c in (ACTION_KEY, STATE_KEY) if c in ds.features]
        action_arr = state_arr = None
        if cols:
            rows = ds.hf_dataset[start:end]
            if ACTION_KEY in rows:
                action_arr = np.array([np.asarray(v, dtype=float) for v in rows[ACTION_KEY]])
            if STATE_KEY in rows:
                state_arr = np.array([np.asarray(v, dtype=float) for v in rows[STATE_KEY]])
        return start, end, action_arr, state_arr

    def _apply_episode(self, start, end, action_arr, state_arr) -> None:
        self._ep_start, self._ep_end = start, end
        self._ep_len = max(1, end - start)
        self._ep_fps = self.dataset.meta.fps or 1
        self._draw_plots(action_arr, state_arr)
        self.frame_slider.configure(from_=0, to=max(0, self._ep_len - 1))
        self.frame_slider.set(0)
        self.play_btn.configure(state="normal")
        self._render_frame(0)

    def _draw_plots(self, action_arr, state_arr) -> None:
        import numpy as np

        self.ax_action.clear()
        self.ax_state.clear()
        self._vline_action = self._vline_state = None

        if action_arr is not None and action_arr.size:
            names = get_feature_names(self.dataset, ACTION_KEY)
            t = np.arange(action_arr.shape[0])
            for j in range(action_arr.shape[1]):
                self.ax_action.plot(
                    t, action_arr[:, j], linewidth=1.0, label=names[j] if j < len(names) else f"a{j}"
                )
            self.ax_action.set_ylabel("action")
            self.ax_action.legend(fontsize=7, loc="best", ncol=max(1, action_arr.shape[1] // 6))
            self._vline_action = self.ax_action.axvline(0, color="black", lw=1.0, alpha=0.5, animated=True)
        else:
            self.ax_action.set_ylabel("action (none)")

        if state_arr is not None and state_arr.size:
            names = get_feature_names(self.dataset, STATE_KEY)
            t = np.arange(state_arr.shape[0])
            for j in range(state_arr.shape[1]):
                self.ax_state.plot(
                    t, state_arr[:, j], linewidth=1.0, label=names[j] if j < len(names) else f"s{j}"
                )
            self.ax_state.set_ylabel("state")
            self.ax_state.set_xlabel("frame")
            self.ax_state.legend(fontsize=7, loc="best", ncol=max(1, state_arr.shape[1] // 6))
            self._vline_state = self.ax_state.axvline(0, color="black", lw=1.0, alpha=0.5, animated=True)
        else:
            self.ax_state.set_ylabel("state (none)")
            self.ax_state.set_xlabel("frame")

        # Full draw to lay out legends, then capture the static background used
        # for blitting during playback (only the vlines move each tick).
        self.mpl_canvas.draw()
        self._capture_blit_background()

    def _capture_blit_background(self) -> None:
        """Snapshot the figure background so playback can blit cheaply.

        We grab the bounding boxes of the two axes (plus their tick/label area)
        instead of the whole figure, so blitting only repaints the plot regions.
        ``animated=True`` artists are skipped during the snapshot automatically.
        """
        import numpy as np  # noqa: F401  (kept for parity with other helpers)

        self._blit_bg = None
        self._blit_bbox = None
        axes = [ax for ax in (self.ax_action, self.ax_state) if ax is not None]
        if not axes:
            return
        # Union bbox of all axes (in display coords).
        bboxes = [ax.get_window_extent() for ax in axes]
        x0 = min(b.x0 for b in bboxes)
        y0 = min(b.y0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)
        from matplotlib.transforms import Bbox

        self._blit_bbox = Bbox.from_extents(x0, y0, x1, y1)
        self._blit_bg = self.mpl_canvas.copy_from_bbox(self._blit_bbox)

    def _build_camera_panes(self) -> None:
        if self.cam_frame_inner is not None:
            self.cam_frame_inner.destroy()
        if self._placeholder_label is not None:
            self._placeholder_label.destroy()
            self._placeholder_label = None
        self.cam_frame_inner = ttk.Frame(self.cam_frame)
        self.cam_frame_inner.pack(fill="both", expand=True)
        self._camera_labels = {}
        cam_keys = list(getattr(self.dataset.meta, "camera_keys", []))
        if not cam_keys:
            ttk.Label(self.cam_frame_inner, text="(no camera features)").pack(expand=True)
            return
        for key in cam_keys:
            col = ttk.Frame(self.cam_frame_inner, borderwidth=1, relief="groove")
            col.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            ttk.Label(col, text=key, anchor="w").pack(fill="x", padx=2, pady=(2, 0))
            lbl = ttk.Label(col, text="(loading…)", anchor="center")
            lbl.pack(fill="both", expand=True, padx=2, pady=2)
            self._camera_labels[key] = lbl

    def _show_frame_images(self, frame_dict: dict[str, object]) -> None:
        """Display decoded camera frames (PIL images) in the camera panes."""
        from PIL import ImageTk

        current_keys = set(getattr(self.dataset.meta, "camera_keys", []))
        if set(self._camera_labels) != current_keys:
            self._build_camera_panes()
        max_w = 480
        for key, lbl in self._camera_labels.items():
            pil = frame_dict.get(key)
            if pil is None:
                continue
            if pil.width > max_w:
                pil = pil.resize((max_w, int(pil.height * max_w / pil.width)))
            photo = ImageTk.PhotoImage(pil)
            lbl.configure(image=photo, text="")
            lbl.image = photo  # prevent GC

    def _update_vlines(self, rel_idx: int, *, blit: bool) -> None:
        """Move the position markers to ``rel_idx``.

        With ``blit=True`` (playback) only the vlines are redrawn over a cached
        background — far cheaper than a full figure redraw. With ``blit=False``
        (manual scrub / single-frame) a normal ``draw_idle`` is used.
        """
        vlines = [v for v in (self._vline_action, self._vline_state) if v is not None]
        for v in vlines:
            v.set_xdata([rel_idx])
        if not blit or self._blit_bg is None or self._blit_bbox is None or not vlines:
            self.mpl_canvas.draw_idle()
            return
        # Blitting: restore static background, redraw only the vlines, flush region.
        self.mpl_canvas.restore_region(self._blit_bg)
        for v in vlines:
            v.axes.draw_artist(v)
        self.mpl_canvas.blit(self._blit_bbox)

    def _render_frame(self, rel_idx: int) -> None:
        """Decode and render a single frame (used on manual scrub)."""
        if self.dataset is None or self._ep_start is None:
            return
        rel_idx = max(0, min(rel_idx, self._ep_len - 1))
        item = self.dataset[self._ep_start + rel_idx]  # decodes videos
        frame_dict = {
            key: frame_to_pil(self.dataset, key, item[key]) for key in self._camera_labels if key in item
        }
        self._show_frame_images(frame_dict)
        self._update_vlines(rel_idx, blit=False)
        self.frame_label_var.set(f"frame {rel_idx}/{self._ep_len - 1}")

    def _on_slider(self, value: str) -> None:
        if self._ep_start is None:
            return
        if self._slider_programmatic:
            return
        self._stop_play()
        self._render_frame(int(float(value)))

    # ── Playback with prefetch + blitting ─────────────────────────────────

    def _decode_frame_for_display(self, abs_idx: int) -> dict[str, object]:
        """Decode one frame off-thread and return ``{cam_key: PIL.Image}``."""
        item = self.dataset[abs_idx]
        return {
            key: frame_to_pil(self.dataset, key, item[key])
            for key in getattr(self.dataset.meta, "camera_keys", [])
            if key in item
        }

    def _start_producer(self, start_rel: int) -> None:
        """Start (or restart) a daemon producer that decodes frames ahead.

        Cancels any previous producer by bumping the generation counter. The new
        producer fills ``_frame_queue`` from ``start_rel`` onward until it reaches
        the episode end or is superseded.
        """
        self._producer_gen += 1
        gen = self._producer_gen
        # Drain any stale frames from a previous producer.
        with suppress(queue.Empty):
            while True:
                self._frame_queue.get_nowait()

        if self.dataset is None or self._ep_start is None:
            return

        def _produce() -> None:
            idx = start_rel
            while idx < self._ep_len and gen == self._producer_gen:
                try:
                    frame_dict = self._decode_frame_for_display(self._ep_start + idx)
                except Exception:  # noqa: BLE001
                    logger.exception("Frame decode failed at %d", idx)
                    return
                if gen != self._producer_gen:
                    return
                try:
                    self._frame_queue.put((idx, frame_dict), timeout=5.0)
                except queue.Full:
                    return
                idx += 1

        self._producer_thread = threading.Thread(target=_produce, daemon=True)
        self._producer_thread.start()

    def toggle_play(self) -> None:
        if self._play_playing:
            self._stop_play()
            return
        if self._ep_start is None:
            return
        self._play_playing = True
        self.play_btn.configure(text="Pause")
        # Cursor = the frame currently on screen. Producer decodes the *next*
        # frames ahead; the consumer pops them in order (no resync needed).
        self._play_cursor = int(float(self.frame_slider.get()))
        if self._play_cursor >= self._ep_len - 1:
            # Already at the end — restart from the beginning.
            self._play_cursor = -1
            self._slider_programmatic = True
            self.frame_slider.set(0)
            self._slider_programmatic = False
        self._start_producer(self._play_cursor + 1)
        self._tick_play()

    def _stop_play(self) -> None:
        self._play_playing = False
        self._producer_gen += 1  # cancel any in-flight producer
        if self._play_after_id is not None:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None
        with suppress(Exception):
            self.play_btn.configure(text="Play")

    def _tick_play(self) -> None:
        if not self._play_playing or self._ep_start is None:
            return
        want = self._play_cursor + 1
        if want >= self._ep_len:
            self._stop_play()
            return
        # Pull the next pre-decoded frame; if the producer hasn't caught up,
        # retry shortly without advancing (no frame dropping).
        try:
            idx, frame_dict = self._frame_queue.get(timeout=2.0)
        except queue.Empty:
            self._play_after_id = self.root.after(15, self._tick_play)
            return
        # Tolerate stale entries from a superseded producer: skip until the one we want.
        if idx != want:
            self._play_after_id = self.root.after(0, self._tick_play)
            return
        self._show_frame_images(frame_dict)
        self._slider_programmatic = True
        self.frame_slider.set(want)
        self._slider_programmatic = False
        self._update_vlines(want, blit=True)
        self.frame_label_var.set(f"frame {want}/{self._ep_len - 1}")
        self._play_cursor = want
        delay = max(1, int(1000 / self._ep_fps))
        self._play_after_id = self.root.after(delay, self._tick_play)

    # ── Edit tab ───────────────────────────────────────────────────────────

    def _on_edit_click(self, event) -> None:
        if self.dataset is None:
            return
        col = self.edit_tree.identify_column(event.x)
        row_id = self.edit_tree.identify_row(event.y)
        if row_id and col == "#0":
            ep = int(row_id)
            if ep in self._edit_marks:
                self._edit_marks.discard(ep)
                self.edit_tree.item(row_id, text="☐")
            else:
                self._edit_marks.add(ep)
                self.edit_tree.item(row_id, text="☑")
            self._refresh_edit_status(self._episodes_rows())

    def _refresh_edit_status(self, rows: list[dict]) -> None:
        if self.dataset is None:
            self.edit_status_var.set("No dataset loaded.")
            return
        frames_by_idx = {r["index"]: r["length"] for r in rows}
        marked = sorted(self._edit_marks)
        rm_frames = sum(frames_by_idx.get(i, 0) for i in marked)
        total = self.dataset.meta.total_frames
        keep = total - rm_frames
        self.edit_status_var.set(
            f"Marked {len(marked)} episodes ({rm_frames} frames) for deletion -> "
            f"{keep}/{total} frames remain."
        )

    def on_delete_marked(self) -> None:
        if self.dataset is None:
            return
        marked = sorted(self._edit_marks)
        if not marked:
            messagebox.showinfo("Delete episodes", "No episodes marked for deletion.")
            return
        new_repo = self.out_repo_var.get().strip()
        new_root = self.out_root_var.get().strip()
        if not new_repo:
            messagebox.showwarning("Delete episodes", "Set an Output repo_id.")
            return
        input_root = Path(self.dataset.root).resolve()
        out_root_resolved = Path(new_root).resolve() if new_root else None
        in_place = out_root_resolved == input_root
        verb = "in-place (original backed up to *_old)" if in_place else "to new dataset"
        if not messagebox.askyesno(
            "Confirm delete",
            f"Delete {len(marked)} episode(s): {marked}\nWrite {verb}:\n"
            f"  repo_id: {new_repo}\n  dir: {new_root}",
        ):
            return

        def work():
            return run_delete_episodes(self.dataset, marked, new_repo or None, new_root or None)

        def done(_new_ds, err):
            self.set_busy(False)
            if err is not None:
                messagebox.showerror("Delete episodes", f"Failed:\n{err}")
                return
            messagebox.showinfo(
                "Delete episodes",
                "Done. Re-open the result from the top bar if you want to preview it.",
            )
            self._edit_marks.clear()
            self._refresh_edit_status(self._episodes_rows())

        logger.info("Deleting episodes %s -> repo_id=%s dir=%s", marked, new_repo, new_root)
        self.run_async(work, done)

    # ── Combine tab ────────────────────────────────────────────────────────

    def on_add_source(self) -> None:
        src = self.merge_src_var.get().strip()
        if not src:
            messagebox.showwarning("Combine", "Enter a repo_id, HF URL, or local path.")
            return
        try:
            repo_id, root = normalize_hub_source(src)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Combine", f"Invalid reference:\n{e}")
            return

        def work():
            meta = load_dataset_meta(repo_id, root)
            return meta.total_episodes, meta.total_frames

        def done(res, err):
            self.set_busy(False)
            if err is not None:
                messagebox.showerror("Combine", f"Could not load metadata:\n{err}")
                return
            eps, frames = res
            self._merge_sources.append((repo_id, root))
            iid = f"{repo_id}@{root}" if root is not None else f"{repo_id}@hub"
            self.merge_tree.insert(
                "", "end", iid=iid, text=repo_id, values=(eps, frames, str(root) if root else "(hub)")
            )
            self.merge_src_var.set("")
            self._refresh_merge_status()
            logger.info("Added source %s (root=%s): %s ep, %s fr", repo_id, root, eps, frames)

        self.run_async(work, done)

    def on_remove_source(self) -> None:
        sel = self.merge_tree.selection()
        if not sel:
            return
        for iid in sel:
            self.merge_tree.delete(iid)
        # Rebuild source list from remaining rows.
        self._merge_sources = []
        for item in self.merge_tree.get_children():
            text = self.merge_tree.item(item, "text")
            root_val = self.merge_tree.set(item, "root")
            root = Path(root_val) if root_val and root_val != "(hub)" else None
            self._merge_sources.append((text, root))
        self._refresh_merge_status()

    def _refresh_merge_status(self) -> None:
        n = len(self._merge_sources)
        self.merge_status_var.set(f"{n} source(s) queued.")
        self.merge_btn.configure(state="normal" if n >= 1 else "disabled")

    def on_merge(self) -> None:
        if not self._merge_sources:
            return
        new_repo = self.out_repo_var.get().strip()
        new_root = self.out_root_var.get().strip()
        if not new_repo:
            messagebox.showwarning("Combine", "Set an Output repo_id.")
            return
        cv = self.concat_videos_var.get()
        cd = self.concat_data_var.get()
        if not messagebox.askyesno(
            "Confirm merge",
            f"Merge {len(self._merge_sources)} dataset(s) into:\n  repo_id: {new_repo}\n  dir: {new_root}\n"
            f"concatenate_videos={cv}, concatenate_data={cd}",
        ):
            return

        sources = list(self._merge_sources)

        def work():
            return run_merge(sources, new_repo, new_root or None, cv, cd)

        def done(merged, err):
            self.set_busy(False)
            if err is not None:
                messagebox.showerror("Combine", f"Merge failed:\n{err}")
                return
            messagebox.showinfo(
                "Combine",
                f"Merged. {merged.meta.total_episodes} episodes, "
                f"{merged.meta.total_frames} frames at\n{merged.root}",
            )

        logger.info("Merging %d sources -> repo_id=%s dir=%s", len(sources), new_repo, new_root)
        self.run_async(work, done)

    # ── Push to Hub ────────────────────────────────────────────────────────

    def on_push_to_hub(self) -> None:
        """Open a push dialog and upload the chosen dataset to the Hugging Face Hub."""
        # Defaults: output dir if it's a dataset, else the loaded dataset's root.
        default_dir = self.out_root_var.get().strip()
        if not (default_dir and (Path(default_dir) / "meta" / "info.json").exists()):
            default_dir = str(self.dataset.root) if self.dataset is not None else ""
        default_repo = self.out_repo_var.get().strip()
        if not default_repo and self.dataset is not None:
            default_repo = self.dataset.repo_id

        dlg = tk.Toplevel(self.root)
        dlg.title("Push to Hugging Face Hub")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, False)

        ttk.Label(dlg, text="Source directory:").grid(row=0, column=0, sticky="w", padx=8, pady=(12, 4))
        dir_var = tk.StringVar(value=default_dir)
        ttk.Entry(dlg, textvariable=dir_var, width=54).grid(row=0, column=1, columnspan=2, sticky="we", padx=4)
        ttk.Button(dlg, text="Browse…", command=lambda: self._browse_into(dlg, dir_var)).grid(
            row=0, column=3, padx=4
        )

        ttk.Label(dlg, text="repo_id:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        repo_var = tk.StringVar(value=default_repo)
        ttk.Entry(dlg, textvariable=repo_var, width=54).grid(row=1, column=1, columnspan=2, sticky="we", padx=4)

        private_var = tk.BooleanVar(value=False)
        push_videos_var = tk.BooleanVar(value=True)
        tag_version_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, text="private repo", variable=private_var).grid(
            row=2, column=0, sticky="w", padx=8, pady=4
        )
        ttk.Checkbutton(dlg, text="push videos", variable=push_videos_var).grid(
            row=2, column=1, sticky="w", padx=4
        )
        ttk.Checkbutton(dlg, text="tag codebase version", variable=tag_version_var).grid(
            row=2, column=2, sticky="w", padx=4
        )

        buttons = ttk.Frame(dlg)
        buttons.grid(row=3, column=0, columnspan=4, pady=(8, 12))

        def _close() -> None:
            dlg.destroy()

        def _do_push() -> None:
            src_dir = dir_var.get().strip()
            repo_id = repo_var.get().strip()
            if not src_dir:
                messagebox.showerror("Push to Hub", "Select a source directory.", parent=dlg)
                return
            if not (Path(src_dir) / "meta" / "info.json").exists():
                messagebox.showerror(
                    "Push to Hub",
                    f"Not a LeRobot dataset (no meta/info.json):\n{src_dir}",
                    parent=dlg,
                )
                return
            if not repo_id or "/" not in repo_id:
                messagebox.showerror(
                    "Push to Hub", "repo_id must look like 'user/dataset_name'.", parent=dlg
                )
                return
            private = private_var.get()
            push_videos = push_videos_var.get()
            tag_version = tag_version_var.get()
            if not messagebox.askyesno(
                "Confirm push",
                f"Upload '{src_dir}' to:\n  https://huggingface.co/datasets/{repo_id}\n"
                f"private={private}, push_videos={push_videos}, tag_version={tag_version}\n\n"
                "Authentication uses your cached 'huggingface-cli login' token.",
                parent=dlg,
            ):
                return
            dlg.destroy()
            self._execute_push(repo_id, src_dir, private, push_videos, tag_version)

        ttk.Button(buttons, text="Cancel", command=_close).pack(side="left", padx=8)
        ttk.Button(buttons, text="Push", command=_do_push).pack(side="left", padx=8)
        dlg.columnconfigure(1, weight=1)
        dlg.columnconfigure(2, weight=1)
        dlg.wait_window()

    def _browse_into(self, dlg: tk.Toplevel, var: tk.StringVar) -> None:
        d = filedialog.askdirectory(title="Select dataset directory", parent=dlg)
        if d:
            var.set(d)

    def _execute_push(
        self, repo_id: str, src_dir: str, private: bool, push_videos: bool, tag_version: bool
    ) -> None:
        logger.info(
            "Pushing %s -> %s (private=%s, push_videos=%s, tag_version=%s)",
            src_dir,
            repo_id,
            private,
            push_videos,
            tag_version,
        )

        def work():
            ds = load_dataset(repo_id, Path(src_dir))
            ds.push_to_hub(
                push_videos=push_videos,
                private=private,
                tag_version=tag_version,
            )
            return repo_id

        def done(rpid, err):
            self.set_busy(False)
            if err is not None:
                messagebox.showerror("Push to Hub", f"Push failed:\n{err}")
                return
            url = f"https://huggingface.co/datasets/{rpid}"
            messagebox.showinfo("Push to Hub", f"Done.\n{url}")
            logger.info("Pushed to %s", url)

        self.run_async(work, done)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        logging.getLogger("lerobot").removeHandler(self._log_handler)
        self._stop_play()


def main() -> None:
    """Launch the dataset editor GUI."""
    import sys

    from lerobot.utils.utils import init_logging

    init_logging()
    try:
        root = tk.Tk()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"Could not create a Tk window ({e}). The dataset editor requires an interactive display.\n"
        )
        sys.exit(2)

    app = DatasetEditorApp(root)

    def on_close() -> None:
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

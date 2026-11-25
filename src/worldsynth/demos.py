"""Recorded robot demos (Isaac Lab / Mimic HDF5) as a WorldSynth dataset, and back.

``import_demos`` turns every demo and camera of an HDF5 file into a clip: ``rgb/`` from ``obs/<camera>``, and
``depth/`` and ``segmentation/`` from ``obs/<camera>_depth`` [mm] and ``obs/<camera>_seg`` (integer ids) when they
were recorded. Labels (masks, 2D boxes) come from the segmentation. The clips then go through the usual Cosmos steps.

``export_demos`` writes a copy of the HDF5 file in which each camera's images are replaced by the Cosmos frames,
with states, actions and everything else unchanged, ready for the same conversion to a training dataset as the
original demos. ``convert_command`` builds the command line of such a converter,
which belongs to the robot project (WorldSynth only runs it).

Only ``h5py`` and the HDF5 layout of Isaac Lab's recorder (``data/<demo>/obs/...``) are assumed; nothing robot or
task specific.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from . import labels as L

logger = logging.getLogger("worldsynth")
MIN_PIXEL_AREA = 16


def parse_seg_names(text: str | None) -> dict[int, str]:
    """Class names for segmentation ids: ``"1=robot,2=table,10-15=can"`` or a JSON file ``{"1": "robot", ...}``."""
    if not text:
        return {}
    path = Path(text)
    if path.suffix == ".json" and path.exists():
        return {int(k): str(v) for k, v in json.loads(path.read_text()).items()}
    names = {}
    for item in text.split(","):
        ids, _, name = item.partition("=")
        lo, _, hi = ids.strip().partition("-")
        for i in range(int(lo), int(hi or lo) + 1):
            names[i] = name.strip()
    return names


def _labels_from_segmentation(seg: np.ndarray, names: dict[int, str], class_ids: dict[str, int]) -> list[dict]:
    objects = []
    for sid in np.unique(seg):
        if sid == 0:
            continue
        mask = seg == sid
        area = int(mask.sum())
        name = names.get(int(sid), f"id_{int(sid)}")
        objects.append(
            {
                "instance_id": int(sid),
                "class": name,
                "class_id": class_ids[name],
                "bbox_2d": L.bbox_from_mask(mask),
                "pixel_area": area,
                "ignore": area < MIN_PIXEL_AREA,
                "mask_rle": L.rle_encode(mask),
            }
        )
    return objects


def import_demos(
    hdf5: str | Path,
    out_dir: str | Path,
    cameras: list[str],
    max_demos: int | None = None,
    seg_names: dict[int, str] | None = None,
    fps: int = 20,
    successful_only: bool = True,
    overwrite: bool = False,
) -> dict:
    """Write the demos of an Isaac Lab HDF5 file as a WorldSynth dataset (one clip per demo and camera).

    Args:
        cameras: Image observations to use, e.g. ``["scene_cam", "wrist_cam"]``.
        max_demos: Use only the first demos (in file order).
        seg_names: Class name per segmentation id (others are called ``id_<n>``).
        successful_only: Skip demos whose ``success`` attribute is false.

    Returns:
        The dataset metadata (also written to ``dataset.json``).
    """
    import h5py

    hdf5, out_dir = Path(hdf5).resolve(), Path(out_dir)
    if out_dir.exists():
        if not overwrite and (out_dir / "dataset.json").exists():
            logger.info("%s exists, keeping finished clips", out_dir)
        elif overwrite:
            shutil.rmtree(out_dir)
    seg_names = dict(seg_names or {})
    clips_root = out_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    resolution, ids_seen, used = None, set(), []
    with h5py.File(hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda n: (len(n), n))
        for demo_name in demos:
            if max_demos is not None and len(used) >= max_demos:
                break
            demo = f["data"][demo_name]
            if successful_only and "success" in demo.attrs and not bool(demo.attrs["success"]):
                continue
            missing = [c for c in cameras if c not in demo["obs"]]
            if missing:
                logger.warning("%s has no %s, skipped", demo_name, missing)
                continue
            used.append(demo_name)
            for cam in cameras:
                rgb = demo["obs"][cam]
                if resolution is None:
                    resolution = [int(rgb.shape[2]), int(rgb.shape[1])]
                if f"{cam}_seg" in demo["obs"]:
                    ids_seen.update(int(i) for i in np.unique(demo["obs"][f"{cam}_seg"][()]) if i)
    if not used:
        raise RuntimeError(f"No demo in {hdf5} has the cameras {cameras}")
    for sid in ids_seen:
        seg_names.setdefault(sid, f"id_{sid}")
    names = sorted(set(seg_names[i] for i in ids_seen))
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    class_ids = {c["name"]: c["id"] for c in categories}

    with h5py.File(hdf5, "r") as f:
        for demo_name in used:
            obs = f["data"][demo_name]["obs"]
            for cam in cameras:
                clip_dir = clips_root / f"{demo_name}__{cam}"
                if (clip_dir / "clip.json").exists():
                    continue
                rgb = obs[cam][()]
                depth = obs[f"{cam}_depth"][()] if f"{cam}_depth" in obs else None
                seg = obs[f"{cam}_seg"][()] if f"{cam}_seg" in obs else None
                for sub in (
                    ("rgb", "labels")
                    + (("depth",) if depth is not None else ())
                    + (("segmentation", "instance") if seg is not None else ())
                ):
                    (clip_dir / sub).mkdir(parents=True, exist_ok=True)
                for t in range(len(rgb)):
                    name = f"{t:06d}"
                    cv2.imwrite(str(clip_dir / "rgb" / f"{name}.png"), cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR))
                    objects = []
                    if depth is not None:
                        d = np.clip(depth[t].reshape(depth.shape[1:3]), 0, 65535).astype(np.uint16)
                        cv2.imwrite(str(clip_dir / "depth" / f"{name}.png"), d)
                    if seg is not None:
                        s = seg[t].reshape(seg.shape[1:3]).astype(np.uint16)
                        cv2.imwrite(str(clip_dir / "segmentation" / f"{name}.png"), s)
                        cv2.imwrite(str(clip_dir / "instance" / f"{name}.png"), s)
                        objects = _labels_from_segmentation(s, seg_names, class_ids)
                    label = {
                        "frame": t,
                        "width": int(rgb.shape[2]),
                        "height": int(rgb.shape[1]),
                        "camera": {},
                        "objects": objects,
                    }
                    (clip_dir / "labels" / f"{name}.json").write_text(json.dumps(label))
                info = {"source": "demos", "hdf5": str(hdf5), "demo": demo_name, "camera": cam,
                        "group": demo_name, "clutter": []}  # fmt: skip
                (clip_dir / "clip.json").write_text(json.dumps(info, indent=1))  # last: marks the clip complete
            logger.info("%s: %d frames x %d cameras", demo_name, len(obs[cameras[0]]), len(cameras))

    meta = {
        "name": out_dir.name,
        "generator": "worldsynth demos",
        "source": {"hdf5": str(hdf5), "cameras": cameras, "demos": used},
        "config": {"dataset": {"fps": fps, "resolution": resolution}},
        "categories": categories,
        "seg_names": {str(k): v for k, v in sorted(seg_names.items())},
        "conventions": {
            "clips": "one clip per demo and camera: clips/<demo>__<camera>",
            "depth": "depth/*.png uint16 millimeters (from obs/<camera>_depth), 0 = invalid",
            "segmentation": "segmentation/*.png uint16 ids (from obs/<camera>_seg), 0 = background",
            "labels": "masks and 2D boxes per segmentation id; no 3D boxes or poses",
        },
        "coco_segmentation": "rle",
    }
    (out_dir / "dataset.json").write_text(json.dumps(meta, indent=2))
    L.build_coco(out_dir, "rgb")
    return meta


def export_demos(
    dataset_dir: str | Path, out_hdf5: str | Path, image_dir: str = "rgb_cosmos", keep_missing: bool = False
) -> list[str]:
    """Copy the source HDF5 of an imported demo dataset with each camera's images replaced by ``image_dir`` frames.

    Args:
        keep_missing: Write a demo as soon as one camera has Cosmos frames; cameras without (e.g. only the scene camera
            was sent to Cosmos) keep their original images. By default a demo is written only when every camera has a
            complete set of frames (e.g. none was rejected by the alignment check).

    Returns:
        The names of the written demos.
    """
    import h5py

    dataset_dir, out_hdf5 = Path(dataset_dir), Path(out_hdf5)
    meta = json.loads((dataset_dir / "dataset.json").read_text())
    src_path, cameras = Path(meta["source"]["hdf5"]), meta["source"]["cameras"]
    written = []
    out_hdf5.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_hdf5.with_suffix(".partial.hdf5")
    with h5py.File(src_path, "r") as src, h5py.File(tmp, "w") as dst:
        for key in src:
            if key != "data":
                src.copy(src[key], dst, name=key)
        data = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            data.attrs[k] = v
        total = 0
        for demo_name in meta["source"]["demos"]:
            demo = src["data"][demo_name]
            frames = {}
            for cam in cameras:
                files = sorted((dataset_dir / "clips" / f"{demo_name}__{cam}" / image_dir).glob("*.png"))
                if len(files) != len(demo["obs"][cam]):
                    continue
                frames[cam] = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files])
            if not frames or (len(frames) != len(cameras) and not keep_missing):
                logger.warning("%s: no complete %s frames for every camera, left out", demo_name, image_dir)
                continue
            src.copy(demo, data, name=demo_name)
            for cam, imgs in frames.items():
                old = demo["obs"][cam]
                if imgs.shape != old.shape:  # Cosmos frames are at the dataset size already; keep the exact shape
                    size = (old.shape[2], old.shape[1])
                    imgs = np.stack([cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in imgs])
                del data[demo_name]["obs"][cam]
                data[demo_name]["obs"].create_dataset(
                    cam, data=imgs.astype(old.dtype), compression=old.compression, compression_opts=old.compression_opts
                )
            total += int(demo.attrs.get("num_samples", len(demo["actions"]) if "actions" in demo else 0))
            written.append(demo_name)
        if "total" in data.attrs:
            data.attrs["total"] = total
        data.attrs["worldsynth_images"] = f"{image_dir} from {dataset_dir}"
    tmp.replace(out_hdf5)
    logger.info("wrote %d demos with %s images to %s", len(written), image_dir, out_hdf5)
    return written


CONVERT_ARGS = "--input_files {inputs} --output_dir {output}"
"""Default arguments of a training-data converter; ``{inputs}`` becomes the HDF5 files, ``{output}`` the folder."""


def export_name(out_hdf5: str | Path, image_dir: str) -> Path:
    """HDF5 file for the frames in ``image_dir``: ``out.hdf5`` for ``rgb_cosmos``, ``out_<look>.hdf5`` for
    ``rgb_cosmos_<look>``."""
    out_hdf5 = Path(out_hdf5)
    look = image_dir.removeprefix("rgb_cosmos").lstrip("_")
    return out_hdf5.with_name(f"{out_hdf5.stem}_{look}{out_hdf5.suffix}") if look else out_hdf5


def convert_command(
    script: str | Path,
    inputs: list[str | Path],
    output: str | Path,
    python: str | None = None,
    args: str = CONVERT_ARGS,
) -> list[str]:
    """Command line of a converter from Isaac Lab HDF5 demos to a training dataset.

    Args:
        script: The converter script, e.g. ``/path/to/convert_training_data.py``.
        python: Interpreter with the converter's dependencies; default: the current one.
        args: Argument template; a ``{inputs}`` word expands to all input files, ``{output}`` to the output folder.
    """
    words = shlex.split(args)
    if "{inputs}" not in words:
        raise ValueError("the converter arguments need an {inputs} word")
    cmd = [str(Path(python).expanduser()) if python else sys.executable, str(Path(script).expanduser())]
    for word in words:
        cmd += [str(p) for p in inputs] if word == "{inputs}" else [word.replace("{output}", str(output))]
    return cmd


def convert_worldsynth_to_lerobot(
    dataset_dir: str | Path,
    output_dir: str | Path,
    image_dir: str = "rgb_cosmos",
    fps: int = 20,
    overwrite: bool = False,
) -> Path:

    import shutil
    import h5py
    import pandas as pd

    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} exists; use overwrite=True")

    meta = json.loads((dataset_dir / "dataset.json").read_text())
    categories = {c["id"]: c["name"] for c in meta["categories"]}

    # Create LeRobot structure
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True)
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    videos_dir = output_dir / "videos" / "chunk-000"
    videos_dir.mkdir(parents=True)

    # Write meta/info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "synthetic",
        "fps": fps,
        "features": {
            "observation.images": {
                "dtype": "video",
                "shape": [720, 1248, 3],  # Default resolution, will be updated from first frame
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [0],  # No robot state for synthetic data
                "names": [],
            },
            "action": {
                "dtype": "float32",
                "shape": [0],  # No actions for synthetic data
                "names": [],
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": ["timestamp"],
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["frame_index"],
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["episode_index"],
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["index"],
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["task_index"],
            },
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    # Write meta/tasks.jsonl
    tasks = [{"task_index": 0, "task": "synthetic_object_detection"}]
    (meta_dir / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))

    # Write meta/episodes.jsonl
    clips = sorted((dataset_dir / "clips").glob("clip_*"))
    episodes = []
    for ep_idx, clip in enumerate(clips):
        clip_meta = json.loads((clip / "clip.json").read_text())
        num_frames = len(list((clip / image_dir).glob("*.png")))
        episodes.append({
            "episode_index": ep_idx,
            "tasks": [0],
            "length": num_frames,
        })
    (meta_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))

    # Write meta/episodes_stats.jsonl
    episodes_stats = []
    for ep_idx, clip in enumerate(clips):
        clip_meta = json.loads((clip / "clip.json").read_text())
        num_frames = len(list((clip / image_dir).glob("*.png")))
        episodes_stats.append({
            "episode_index": ep_idx,
            "stats": {
                "observation.images": {"min": 0, "max": 255, "mean": 127.5, "std": 73.5, "count": num_frames},
            },
        })
    (meta_dir / "episodes_stats.jsonl").write_text("\n".join(json.dumps(e) for e in episodes_stats))

    # Process each clip -> episode parquet
    # Update image shape from first frame
    first_clip = clips[0] if clips else None
    if first_clip:
        first_frame = sorted((first_clip / image_dir).glob("*.png"))
        logger.info("Looking for frames in %s, found %d", first_clip / image_dir, len(first_frame))
        if first_frame:
            import cv2
            first_img = cv2.imread(str(first_frame[0]))
            h, w = first_img.shape[:2]
            info["features"]["observation.images"]["shape"] = [h, w, 3]
            # Rewrite info.json with correct shape
            (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    for ep_idx, clip in enumerate(clips):
        frames = sorted((clip / image_dir).glob("*.png"))
        labels = sorted((clip / "labels").glob("*.json"))
        logger.info("Clip %s: %d frames, %d labels", clip.name, len(frames), len(labels))

        # Create video
        import cv2
        if frames:
            first = cv2.imread(str(frames[0]))
            h, w = first.shape[:2]
            video_path = videos_dir / f"episode_{ep_idx:06d}.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            if not writer.isOpened():
                logger.error("Cannot create video for episode %d", ep_idx)
            else:
                for f in frames:
                    writer.write(cv2.imread(str(f)))
                writer.release()
                logger.debug("Created video: %s", video_path)

        # Create parquet
        rows = []
        for frame_idx, (img_path, label_path) in enumerate(zip(frames, labels)):
            label = json.loads(label_path.read_text())
            rows.append({
                "observation.images": str(video_path.relative_to(output_dir)).replace(".mp4", ".avi"),
                "observation.state": np.array([], dtype=np.float32),
                "action": np.array([], dtype=np.float32),
                "timestamp": np.array([frame_idx / fps], dtype=np.float32),
                "frame_index": np.array([frame_idx], dtype=np.int64),
                "episode_index": np.array([ep_idx], dtype=np.int64),
                "index": np.array([frame_idx], dtype=np.int64),
                "task_index": np.array([0], dtype=np.int64),
            })
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

    logger.info("Converted %d clips to LeRobot format at %s", len(clips), output_dir)
    return output_dir

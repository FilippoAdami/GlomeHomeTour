#
# Turns a finished COLMAP pass into a scene laid out the way the gaussian
# splatting tooling expects it.
#
# The keyframes are the frames COLMAP actually registered in sparse/0/ - i.e.
# the ones that survived feature matching and triangulation, and the only ones
# it would use for a calibration pass anyway. Frames it dropped have no tracks,
# so they carry no geometry and only cost training time.
#
# Layout afterwards (images_all/ holds every original frame, untouched):
#
#   images/      keyframes, full resolution   (hardlinks, so free)
#   images_2/    keyframes at 1/2
#   images_4/    keyframes at 1/4
#   images_8/    keyframes at 1/8
#   transforms_keyframes.json                 second starting point
#
# cameras.txt keeps its full-resolution intrinsics: the 2DGS loader derives
# FovX/FovY from focal/width, which are angles and so resolution-independent,
# and sizes each camera from the image file it actually loads. That is why
# training with --images images_4 needs no change to the sparse model.
#

import os
import json
import shutil
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

LEVELS = (2, 4, 8)
ALL_IMAGES_DIRNAME = "images_all"

# A Sampson filter that wants to drop more than this is not finding a handful of
# broken frames, it is telling you the whole model is wrong. Report, don't apply -
# same call as convert_transforms_to_colmap's rotation-drift refusal.
MAX_SAMPSON_REJECT_FRAC = 0.2


def registered_names(sparse_dir):
    """Basenames of the images present in a COLMAP model's images.txt."""
    from colmap_diagnostics import parse_images_txt

    images = parse_images_txt(os.path.join(sparse_dir, "images.txt"))
    # Zoom-staged scenes register images as "cam_0/frame_00001.jpg".
    return {os.path.basename(name) for name in images}


def sampson_rejects(db_path, sparse_dir, max_px):
    """Frames whose pose disagrees with the raw verified matches, name -> px.

    Registering is not the same as being well posed: a frame next to a tracking
    break gets a pose the matches do not support. Drift does not catch these -
    it measures disagreement with ARCore, which says nothing about correctness -
    so this is the check that does.
    """
    from colmap_diagnostics import parse_cameras_txt, parse_images_txt, per_frame_sampson

    cams = parse_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
    images = parse_images_txt(os.path.join(sparse_dir, "images.txt"))
    per_frame = per_frame_sampson(db_path, cams, images)
    return {os.path.basename(name): err
            for name, err in per_frame.items() if err > max_px}


def filter_by_sampson(keep, rejects, max_reject_frac=MAX_SAMPSON_REJECT_FRAC):
    """Drop rejects from keep, unless there are too many of them to be credible.

    Returns (kept, refused). Kept unchanged when refused.
    """
    if not rejects:
        return keep, False
    if len(rejects) > max_reject_frac * len(keep):
        return keep, True
    return keep - set(rejects), False


def _resize_one(args):
    src, dst, divisor, quality = args
    with Image.open(src) as im:
        w, h = im.size
        im.resize((max(1, round(w / divisor)), max(1, round(h / divisor))),
                  Image.LANCZOS).save(dst, quality=quality)


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)  # different filesystem, or a filesystem without hardlinks


def prune_model(sparse_dir, names):
    """Delete images from the COLMAP model so it matches the exported folders.

    Not optional: the 2DGS loader opens every image the model lists, so a model
    still naming frames that images/ no longer holds crashes it at startup.
    image_deleter also drops their observations and prunes the tracks that leaves
    empty, which hand-editing images.txt would not.
    """
    from colmap_diagnostics import parse_images_txt

    model_names = [name for name in parse_images_txt(os.path.join(sparse_dir, "images.txt"))
                   if os.path.basename(name) in names]
    if not model_names:
        return

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(sorted(model_names)) + "\n")
        list_path = f.name
    try:
        for cmd in (["colmap", "image_deleter",
                     "--input_path", sparse_dir, "--output_path", sparse_dir,
                     "--image_names_path", list_path],
                    # image_deleter writes .bin only; the diagnostics read .txt.
                    ["colmap", "model_converter",
                     "--input_path", sparse_dir, "--output_path", sparse_dir,
                     "--output_type", "TXT"]):
            from Utilities.pipeline_paths import subprocess_env
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           env=subprocess_env({"QT_QPA_PLATFORM": "xcb"}))
    finally:
        os.unlink(list_path)
    print(f"  pruned {len(model_names)} image(s) from {sparse_dir}")


def export_keyframes(source_path, images_dir, transforms, sparse_dir, quality=95,
                     max_sampson_px=None, db_path=None):
    """Write images/ + images_2/4/8 + transforms_keyframes.json for the keyframes.

    images_dir is the folder holding every original frame. It is renamed to
    images_all/ rather than thinned, so nothing the capture produced is lost and
    a re-run can start over from the full set.

    max_sampson_px additionally drops frames that registered but whose pose the
    raw matches do not support; it needs the COLMAP database.
    """
    registered = registered_names(sparse_dir)
    keep, rejected = registered, {}

    if max_sampson_px and db_path and os.path.exists(db_path):
        rejects = sampson_rejects(db_path, sparse_dir, max_sampson_px)
        keep, refused = filter_by_sampson(registered, rejects)
        if refused:
            print(f"\n{len(rejects)}/{len(registered)} frames exceed {max_sampson_px} px "
                  "Sampson error. That is too many to be a handful of bad poses - the model "
                  "itself is suspect, so they are reported rather than quietly dropped.")
        elif rejects:
            rejected = rejects
            worst = sorted(rejects.items(), key=lambda kv: -kv[1])
            print(f"\nSampson filter: dropping {len(rejects)} badly posed frame(s) "
                  f"above {max_sampson_px} px")
            for name, err in worst[:8]:
                print(f"    {name}  {err:8.2f} px")
            if len(worst) > 8:
                print(f"    ... and {len(worst) - 8} more")
    elif max_sampson_px:
        print(f"\n[keyframes] no database at {db_path}, skipping the Sampson filter")

    frames = [f for f in transforms["frames"]
              if os.path.basename(f["file_path"]) in keep]
    if not frames:
        raise RuntimeError(
            f"None of the {len(transforms['frames'])} frames in transforms.json match "
            f"the images registered in {sparse_dir}. Nothing to export.")

    if rejected:
        prune_model(sparse_dir, rejected)

    all_dir = os.path.join(source_path, ALL_IMAGES_DIRNAME)
    if not os.path.exists(all_dir):
        os.rename(images_dir, all_dir)

    names = [os.path.basename(f["file_path"]) for f in frames]
    jobs = []
    for divisor in (1,) + LEVELS:
        out_dir = images_dir if divisor == 1 else os.path.join(source_path, f"images_{divisor}")
        shutil.rmtree(out_dir, ignore_errors=True)  # a re-run must not leave dropped frames behind
        os.makedirs(out_dir)
        for name in names:
            src, dst = os.path.join(all_dir, name), os.path.join(out_dir, name)
            if divisor == 1:
                _link_or_copy(src, dst)
            else:
                jobs.append((src, dst, divisor, quality))

    # PIL drops the GIL inside decode/encode, so threads are enough here.
    with ThreadPoolExecutor() as pool:
        list(pool.map(_resize_one, jobs))

    out_path = os.path.join(source_path, "transforms_keyframes.json")
    with open(out_path, "w") as f:
        json.dump({**transforms, "frames": frames}, f, indent=2)

    total = len(transforms["frames"])
    unregistered = total - len(registered)
    print(f"\nKeyframes: {len(frames)}/{total} frames kept "
          f"({unregistered} unregistered by COLMAP, {len(rejected)} badly posed)")
    print(f"  {images_dir}/ + images_2/4/8 written, originals kept in {all_dir}/")
    print(f"  {out_path}")
    return frames

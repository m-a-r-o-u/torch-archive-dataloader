import argparse
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(input_root: Path) -> List[Path]:
    return sorted([
        path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ])


def _group_image_paths(
    input_root: Path,
    image_paths: List[Path],
    group_depth: int,
) -> DefaultDict[Tuple[str, ...], List[Tuple[Path, Path]]]:
    groups: DefaultDict[Tuple[str, ...], List[Tuple[Path, Path]]] = defaultdict(list)
    for image_path in image_paths:
        relative_path = image_path.relative_to(input_root)
        parts = relative_path.parts
        if group_depth > 0 and len(parts) <= group_depth:
            raise RuntimeError(
                "Image path has fewer path components than --group-depth: "
                f"{relative_path} (group-depth={group_depth})"
            )

        group_key = parts[:group_depth]
        arcname = Path(*parts[group_depth:]) if group_depth > 0 else relative_path
        groups[group_key].append((image_path, arcname))

    return groups


def _write_group_archives(
    grouped_samples: Dict[Tuple[str, ...], List[Tuple[Path, Path]]],
    output_root: Path,
    num_shards: int,
) -> None:
    for group_key in sorted(grouped_samples):
        samples = grouped_samples[group_key]
        group_output_root = output_root.joinpath(*group_key)
        group_output_root.mkdir(parents=True, exist_ok=True)

        shard_size = math.ceil(len(samples) / num_shards)
        group_name = "/".join(group_key) if group_key else "<root>"

        print()
        print(f"Group: {group_name}")
        print(f"  Images: {len(samples)}")
        print(f"  Output: {group_output_root}")
        print(f"  Max shards: {num_shards}")

        written_shards = 0
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, len(samples))
            if start >= len(samples):
                break

            shard_path = group_output_root / f"shard-{shard_idx:05d}.tar"
            with tarfile.open(shard_path, "w") as tar:
                for image_path, arcname in samples[start:end]:
                    tar.add(image_path, arcname=str(arcname))

            written_shards += 1
            print(f"  Wrote {shard_path.name} with {end - start} image(s)")

        print(f"  Done. Wrote {written_shards} shard(s).")


def build_archives(input_root: Path, output_root: Path, num_shards: int, group_depth: int) -> None:
    image_paths = _collect_images(input_root)

    if not image_paths:
        raise RuntimeError(f"No images found in {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    grouped_samples = _group_image_paths(
        input_root=input_root,
        image_paths=image_paths,
        group_depth=group_depth,
    )

    print(f"Found {len(image_paths)} images under {input_root}")
    print(f"Detected {len(grouped_samples)} group(s) with --group-depth={group_depth}")
    print(f"Output root: {output_root}")

    _write_group_archives(
        grouped_samples=grouped_samples,
        output_root=output_root,
        num_shards=num_shards,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a dataset tree from loose files into tar shards. Supports "
            "both single-split conversion and multi-group layouts like "
            "ImageNet-C corruption/severity directories."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to one loose-file split, e.g. /path/to/imagenet-c/fog/5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to output shard directory, e.g. /path/to/imagenet-c-archives/fog/5",
    )
    parser.add_argument(
        "--group-depth",
        type=int,
        default=0,
        help=(
            "Number of leading path components to treat as grouping folders. "
            "Examples: 0 for a single split root; 2 for ImageNet-C full dataset "
            "(corruption/severity)."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=16,
        help="Maximum number of tar shards to generate (default: 16)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if args.group_depth < 0:
        raise ValueError("--group-depth must be >= 0")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_root}")

    build_archives(
        input_root=args.input_root,
        output_root=args.output_root,
        num_shards=args.num_shards,
        group_depth=args.group_depth,
    )


if __name__ == "__main__":
    main()

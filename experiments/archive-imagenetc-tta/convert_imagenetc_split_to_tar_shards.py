import argparse
import math
import tarfile
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_archives(input_root: Path, output_root: Path, num_shards: int) -> None:
    image_paths = sorted([
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ])

    if not image_paths:
        raise RuntimeError(f"No images found in {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    shard_size = math.ceil(len(image_paths) / num_shards)

    print(f"Found {len(image_paths)} images")
    print(f"Writing up to {num_shards} shard(s) to {output_root}")

    written_shards = 0
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(image_paths))
        if start >= len(image_paths):
            break

        shard_path = output_root / f"shard-{shard_idx:05d}.tar"
        with tarfile.open(shard_path, "w") as tar:
            for image_path in image_paths[start:end]:
                arcname = image_path.relative_to(input_root)
                tar.add(image_path, arcname=str(arcname))

        written_shards += 1
        print(f"Wrote {shard_path} with {end - start} image(s)")

    print(f"Done. Wrote {written_shards} shard(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one ImageNet-C corruption/severity split from loose files "
            "into tar shards while preserving class-relative paths."
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
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_root}")

    build_archives(
        input_root=args.input_root,
        output_root=args.output_root,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()

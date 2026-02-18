import argparse
import importlib

EXAMPLE_TO_MODULE = {
    "individual-files": "examples.minimal_imagenetc_tta_example",
    "tar-archive": "examples.imagenetc_tta_tar_archive_example",
    "memory-cache": "examples.imagenetc_tta_memory_cache_example",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run one of the ImageNet-C loading examples.")
    parser.add_argument(
        "--mode",
        choices=EXAMPLE_TO_MODULE.keys(),
        default="individual-files",
        help="Dataset storage/loading strategy to run.",
    )
    args = parser.parse_args()

    module_name = EXAMPLE_TO_MODULE[args.mode]
    module = importlib.import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()

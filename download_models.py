from huggingface_hub import snapshot_download
import os
import argparse
from pathlib import Path

MODEL_REPOS = {
    'qwen3.8': 'Qwen/Qwen3.8-27B',
    'gemma4': 'google/gemma-4-31B-it',
    'muse': 'meta-models/Muse-Glimmer-30B',
}
MODEL_HASHES = {
    'qwen3.8': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
    'gemma4': '842da3794eaa0b77d5f08bae87a17459d91ff475',
    'muse': 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()

    for model, version_hash in MODEL_HASHES.items():
        model_dir = args.model_root / model / f"{model}-{version_hash}"
        model_dir.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=MODEL_REPOS[model],
            revision=version_hash,
            local_dir=model_dir,
        )
        print(f"Downloaded {model} to {model_dir}")


if __name__ == "__main__":
    main()

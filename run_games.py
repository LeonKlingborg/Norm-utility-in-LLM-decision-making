import argparse
from pathlib import Path
from vLLM_setup import MODEL_SETTINGS, run_games
from prompt_templates import TEMPLATES

def main(
        model_root,
        data_storage_path,
        model = 'qwen3.8',
        condition = 'with_belief_with_util',
        B = 3,
        C = 1.5,
        public_belief_proportion = 50,
        temperature = 1.0,
        sample_size = 100
):

    df = run_games(
        model=model,
        model_root=model_root,
        condition=condition,
        benefit=B,
        cost=C,
        public_belief_proportion=public_belief_proportion,
        temperature=temperature,
        sample_size=sample_size,
    )

    data_storage_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_storage_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_SETTINGS, default="qwen3.8")
    parser.add_argument("--condition", choices=TEMPLATES, default="with_belief_with_util")
    args = parser.parse_args()

    main(
        model_root=args.model_root,
        data_storage_path=args.output,
        model=args.model,
        condition=args.condition,
    )

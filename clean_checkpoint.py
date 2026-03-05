import json
from pathlib import Path


def main():
    """Clean parts of the v5 checkpoint so specific items can be re-run.

    The checkpoint keys are of the form:
        "{model}|{dataset}|{prompt_type}|{idx}"

    For the current v5 experiment, we want to remove entries corresponding to
    the *first* incremental file for a particular (model, dataset, prompt).
    By default this targets:

        model       = "llama3.1"
        dataset     = "danish_metaphors_v5"
        prompt_type = "met_v1"
        idx range   = [0, 99]

    Adjust these constants below if you need to clean a different span.
    """

    # Checkpoint for v5 experiment (see config.yaml: name + output_dir)
    ckpt_path = Path("results/v5") / "Danish Metaphor Benchmark v5_checkpoint.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Target spans to remove. Each entry describes a contiguous index
    # range for a specific (model, dataset, prompt_type).
    #
    # For the current corrupted incremental file, we need to redo:
    #   1) llama3.1 | danish_metaphors_v5 | met_v1 | idx  99–913
    #   2) llama3.1 | danish_metaphors_v5 | met_v2 | idx   0–913
    #   3) gemma2   | danish_metaphors_v5 | met_v1 | idx   0–866
    TARGET_SPANS = [
        {
            "model": "llama3.1",
            "dataset": "danish_metaphors_v5",
            "prompt": "met_v1",
            "idx_min": 99,
            "idx_max": 913,
        },
        {
            "model": "llama3.1",
            "dataset": "danish_metaphors_v5",
            "prompt": "met_v2",
            "idx_min": 0,
            "idx_max": 913,
        },
        {
            "model": "gemma2",
            "dataset": "danish_metaphors_v5",
            "prompt": "met_v1",
            "idx_min": 0,
            "idx_max": 866,
        },
    ]

    print("Cleaning checkpoint:", ckpt_path)
    print("Target spans:")
    for span in TARGET_SPANS:
        print(
            "  - model=%s, dataset=%s, prompt=%s, idx=[%d, %d]"
            % (
                span["model"],
                span["dataset"],
                span["prompt"],
                span["idx_min"],
                span["idx_max"],
            )
        )

    # Backup first
    backup_path = ckpt_path.with_suffix(".backup.json")
    backup_path.write_text(ckpt_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup written to: {backup_path}")

    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    items = data.get("processed_items", [])
    before = len(items)

    def should_remove(key: str) -> bool:
        parts = key.split("|")
        if len(parts) != 4:
            return False
        model, dataset, prompt, idx_str = parts
        try:
            idx = int(idx_str)
        except ValueError:
            return False

        for span in TARGET_SPANS:
            if (
                model == span["model"]
                and dataset == span["dataset"]
                and prompt == span["prompt"]
                and span["idx_min"] <= idx <= span["idx_max"]
            ):
                return True
        return False

    filtered = [k for k in items if not should_remove(k)]
    after = len(filtered)
    removed = before - after

    data["processed_items"] = filtered
    ckpt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total processed_items before: {before}")
    print(f"Total processed_items after : {after}")
    print(f"Removed {removed} entries in the specified span.")


if __name__ == "__main__":
    main()
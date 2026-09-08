"""
Download HotpotQA dev set using the HuggingFace datasets library.
Run: python improvement_files/datasets/download_hotpotqa.py
"""
import os

def main():
    from datasets import load_dataset

    base = os.path.dirname(os.path.abspath(__file__))

    print("Downloading HotpotQA dev set (distractor setting)...")
    print("This is ~113k multi-hop QA pairs from Wikipedia.")
    print()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    out = os.path.join(base, "hotpotqa_dev.json")
    ds.to_json(out)

    print(f"Saved {len(ds)} examples to {out}")
    print(f"File size: {os.path.getsize(out) / (1024*1024):.1f} MB")
    print()
    print("Sample entry keys:", list(ds[0].keys()))
    print("Sample question:", ds[0]['question'])
    print("Sample answer:", ds[0]['answer'])
    print("Question type:", ds[0]['type'])

if __name__ == "__main__":
    main()

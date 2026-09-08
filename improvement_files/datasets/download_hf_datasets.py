"""
Download datasets from HuggingFace for AHRAG improvement.
Run: python improvement_files/datasets/download_hf_datasets.py
"""
import os
import json

def main():
    from datasets import load_dataset

    base = os.path.dirname(os.path.abspath(__file__))

    # 1. ConflictQA — knowledge conflict evaluation
    print("[1/4] Downloading ConflictQA...")
    try:
        ds = load_dataset("osunlp/ConflictQA", "conflictQA-popqa-chatgpt", trust_remote_code=True)
        out = os.path.join(base, "conflictqa_popqa.json")
        ds["test"].to_json(out)
        print(f"  -> Saved to {out} ({len(ds['test'])} examples)")
    except Exception as e:
        print(f"  -> Failed: {e}")

    # 2. PolicyQA — privacy policy question answering
    print("[2/4] Downloading PolicyQA...")
    try:
        ds = load_dataset("alzoubi36/policy_qa", trust_remote_code=True)
        out = os.path.join(base, "policyqa_train.json")
        ds["train"].to_json(out)
        print(f"  -> Saved to {out} ({len(ds['train'])} examples)")
    except Exception as e:
        print(f"  -> Failed: {e}")

    # 3. HR Policies QA Dataset
    print("[3/4] Downloading HR Policies QA Dataset...")
    try:
        ds = load_dataset("strova-ai/hr-policies-qa-dataset", trust_remote_code=True)
        split_name = list(ds.keys())[0]
        out = os.path.join(base, "hr_policies_qa.json")
        ds[split_name].to_json(out)
        print(f"  -> Saved to {out} ({len(ds[split_name])} examples)")
    except Exception as e:
        print(f"  -> Failed: {e}")

    # 4. FinanceBench from HuggingFace
    print("[4/4] Downloading FinanceBench from HuggingFace...")
    try:
        ds = load_dataset("PatronusAI/financebench", trust_remote_code=True)
        split_name = list(ds.keys())[0]
        out = os.path.join(base, "financebench_hf.json")
        ds[split_name].to_json(out)
        print(f"  -> Saved to {out} ({len(ds[split_name])} examples)")
    except Exception as e:
        print(f"  -> Failed: {e}")

    print("\nDone! Check the datasets/ folder for downloaded files.")


if __name__ == "__main__":
    main()

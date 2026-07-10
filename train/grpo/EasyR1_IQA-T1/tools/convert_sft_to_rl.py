# Copyright 2024 ThinkVis-IQA Project
# Convert SFT data format to EasyR1 RL format

import json
import argparse
from pathlib import Path


def convert_sft_to_rl(input_file: str, output_file: str):
    """
    Convert SFT training data to EasyR1 RL format.
    
    SFT format:
    {
        "id": "xxx",
        "gt_score": 3.46,
        "image": ["koniq/xxx.jpg", "tools/xxx/xxx.png", ...],
        "conversations": [
            {"from": "human", "value": "...\\n<image>"},
            {"from": "gpt", "value": "<think>...</think>\\n<answer_start>{\"score\": X.XX}<answer_end>"}
        ]
    }
    
    EasyR1 RL format:
    {
        "problem": "...\\n<image>",
        "answer": "3.46",
        "images": ["koniq/xxx.jpg", "tools/xxx/xxx.png", ...]
    }
    """
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    
    print(f"Total samples in SFT data: {len(sft_data)}")
    
    rl_data = []
    skipped = 0
    
    for item in sft_data:
        try:
            # Extract fields
            gt_score = item.get('gt_score')
            images = item.get('image', [])
            conversations = item.get('conversations', [])
            
            # Find user question (human turn)
            problem = None
            for conv in conversations:
                if conv.get('from') == 'human':
                    problem = conv.get('value', '')
                    break
            
            if problem is None or gt_score is None:
                skipped += 1
                continue
            
            # Create RL format entry
            rl_entry = {
                "problem": problem,
                "answer": str(gt_score),
                "images": images
            }
            rl_data.append(rl_entry)
            
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            skipped += 1
            continue
    
    print(f"Converted samples: {len(rl_data)}")
    print(f"Skipped samples: {skipped}")
    
    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rl_data, f, ensure_ascii=False, indent=2)
    
    print(f"Output saved to: {output_file}")
    
    # Print sample
    if rl_data:
        print("\n=== Sample output ===")
        print(json.dumps(rl_data[0], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Convert SFT data to EasyR1 RL format')
    parser.add_argument('--input', type=str, required=True, help='Input SFT JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output RL JSON file')
    
    args = parser.parse_args()
    convert_sft_to_rl(args.input, args.output)


if __name__ == '__main__':
    main()

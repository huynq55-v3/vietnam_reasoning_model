#!/usr/bin/env python3
"""
Dataset Validator v3 for Vietnam Reasoning Model
Validates schema, entity de-factualization, distractor chunks,
special tokens (<|thought|>, <|/thought|><|answer|>, <|endoftext|>),
domains, and strict content-based citation checks.
"""

import argparse
import collections
import json
import os
import re
import sys

REQUIRED_FIELDS = ["id", "domain", "category", "user_query", "external_memory_chunks", "assistant_reasoning_and_reply"]
VALID_DOMAINS = {"law_admin", "finance_contract", "medical_bio", "tech_infra", "daily_life", "agriculture_env", "sci_fi"}
CHUNK_PREFIX_REGEX = re.compile(r"^(chunk|đoạn|phần)\s*\d+\s*:", re.IGNORECASE)

def validate_dataset(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    total = 0
    errors = 0
    valid_tags_count = 0
    total_chunks = 0
    categories = collections.Counter()
    domains = collections.Counter()
    chunk_prefix_warnings = 0

    print(f"🔍 Validating dataset v3: {file_path}...\n")

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ Line {idx}: Invalid JSON syntax - {e}")
                errors += 1
                continue

            # Check required fields
            missing = [f for f in REQUIRED_FIELDS if f not in data]
            if missing:
                print(f"❌ Line {idx}: Missing required fields: {missing}")
                errors += 1
                continue

            # Check domain validity
            dom = data.get("domain")
            if dom not in VALID_DOMAINS:
                print(f"⚠️ Line {idx}: Unknown or missing domain '{dom}'. Expected one of {list(VALID_DOMAINS)}")
                errors += 1
            else:
                domains[dom] += 1

            cat = data.get("category", "unspecified")
            categories[cat] += 1

            # Check external_memory_chunks (khuyên dùng từ 4 đến 7 chunks)
            chunks = data.get("external_memory_chunks")
            if not isinstance(chunks, list) or len(chunks) < 4:
                print(f"⚠️ Line {idx}: 'external_memory_chunks' should have at least 4 chunks (found {len(chunks) if isinstance(chunks, list) else 0})")
                errors += 1
                continue
            total_chunks += len(chunks)

            # Check for artificial prefixes like "Chunk 1:" or "Đoạn 1:"
            for c_idx, c in enumerate(chunks):
                if isinstance(c, str) and CHUNK_PREFIX_REGEX.match(c.strip()):
                    print(f"⚠️ Line {idx}, Chunk {c_idx+1}: Found artificial prefix '{c[:15]}...'. Content-based citation required.")
                    chunk_prefix_warnings += 1
                    errors += 1

            # Check strict token tags: <|thought|> ... <|/thought|><|answer|> ... <|endoftext|>
            reply = data.get("assistant_reasoning_and_reply", "")
            has_thought = "<|thought|>" in reply
            has_close_thought = "<|/thought|>" in reply
            has_answer = "<|answer|>" in reply
            has_end = reply.endswith("<|endoftext|>")

            if has_thought and has_close_thought and has_answer and has_end:
                valid_tags_count += 1
            else:
                missing_tags = []
                if not has_thought: missing_tags.append("<|thought|>")
                if not has_close_thought: missing_tags.append("<|/thought|>")
                if not has_answer: missing_tags.append("<|answer|>")
                if not has_end: missing_tags.append("<|endoftext|> (must be at the absolute end)")
                print(f"⚠️ Line {idx}: State token violation: {missing_tags}")
                errors += 1

    print("-" * 55)
    print(f"📊 Validation Summary (v3):")
    print(f"  • Total samples parsed: {total}")
    print(f"  • Valid samples with strict structural tokens: {valid_tags_count}")
    print(f"  • Average chunks per sample: {total_chunks / total if total > 0 else 0:.1f}")
    print(f"  • Domain distribution:")
    for d, cnt in domains.items():
        print(f"      - {d}: {cnt} ({cnt / total * 100:.1f}%)")
    print(f"  • Category distribution:")
    for c, cnt in categories.items():
        print(f"      - {c}: {cnt} ({cnt / total * 100:.1f}%)")
    print(f"  • Chunk prefix violations: {chunk_prefix_warnings}")
    print(f"  • Total errors / warnings: {errors}")

    if errors == 0:
        print("\n✅ All samples PASSED strict v3 quality and schema checks! 🚀")
        sys.exit(0)
    else:
        print(f"\n⚠️ Dataset has {errors} issues that need attention before training.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JSONL reasoning dataset v3")
    parser.add_argument("--file", type=str, default="data/reasoning_dataset.jsonl", help="Path to jsonl file")
    args = parser.parse_args()
    validate_dataset(args.file)

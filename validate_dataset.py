#!/usr/bin/env python3
"""
Dataset Validator & Shuffle Utility for Vietnam Reasoning Model
Validates schema, domains, categories, chunks count, absence of artificial prefixes,
and strict single-occurrence state tokens (<|thought|>, <|/thought|><|answer|>, <|endoftext|>).
Includes a --shuffle flag to randomize external_memory_chunks to eliminate positional bias.
"""

import argparse
import collections
import json
import os
import random
import re
import sys

REQUIRED_FIELDS = [
    "id",
    "domain",
    "category",
    "user_query",
    "external_memory_chunks",
    "assistant_reasoning_and_reply",
]
VALID_DOMAINS = {
    "law_admin",
    "finance_contract",
    "medical_bio",
    "tech_infra",
    "daily_life",
    "agriculture_env",
    "sci_fi",
}
VALID_CATEGORIES = {
    "multi_hop_deduction",
    "boundary_math",
    "counterfactual_rule",
    "negative_refusal",
    "conflict_resolution",
}
CHUNK_PREFIX_REGEX = re.compile(r"^(chunk|đoạn|phần)\s*\d+\s*:", re.IGNORECASE)

# Cấu trúc chuẩn: <|thought|>...<|/thought|><|answer|>...<|endoftext|>
STRICT_FORMAT_REGEX = re.compile(
    r"^<\|thought\|>\n?.+?\n?<\|/thought\|><|answer\|>\n?.+?<\|endoftext\|>$",
    re.DOTALL,
)


def validate_and_process(file_path: str, shuffle_chunks: bool = False):
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file {file_path}")
        sys.exit(1)

    total = 0
    errors = 0
    valid_tags_count = 0
    total_chunks = 0
    categories = collections.Counter()
    domains = collections.Counter()
    chunk_prefix_warnings = 0
    records = []

    print(f"🔍 Đang kiểm tra dataset: {file_path} (Shuffle chunks: {'BẬT' if shuffle_chunks else 'TẮT'})...\n")

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ Dòng {idx}: Lỗi cú pháp JSON - {e}")
                errors += 1
                continue

            # 1. Kiểm tra các trường bắt buộc
            missing = [f for f in REQUIRED_FIELDS if f not in data]
            if missing:
                print(f"❌ Dòng {idx}: Thiếu các trường bắt buộc: {missing}")
                errors += 1
                continue

            # 2. Kiểm tra domain hợp lệ
            dom = data.get("domain")
            if dom not in VALID_DOMAINS:
                print(f"⚠️ Dòng {idx} ({data.get('id')}): Domain không hợp lệ '{dom}'. Hợp lệ: {sorted(VALID_DOMAINS)}")
                errors += 1
            else:
                domains[dom] += 1

            # 3. Kiểm tra category hợp lệ
            cat = data.get("category", "unspecified")
            if cat not in VALID_CATEGORIES:
                print(f"⚠️ Dòng {idx} ({data.get('id')}): Category không hợp lệ '{cat}'. Hợp lệ: {sorted(VALID_CATEGORIES)}")
                errors += 1
            else:
                categories[cat] += 1

            # 4. Kiểm tra external_memory_chunks (4 đến 8 chunks)
            chunks = data.get("external_memory_chunks")
            if not isinstance(chunks, list) or not (4 <= len(chunks) <= 8):
                print(f"⚠️ Dòng {idx} ({data.get('id')}): 'external_memory_chunks' phải chứa 4-8 chunks (thực tế: {len(chunks) if isinstance(chunks, list) else 0})")
                errors += 1
                continue
            total_chunks += len(chunks)

            # 5. Kiểm tra tiền tố nhân tạo
            for c_idx, c in enumerate(chunks):
                if isinstance(c, str) and CHUNK_PREFIX_REGEX.match(c.strip()):
                    print(f"⚠️ Dòng {idx}, Chunk {c_idx+1}: Chứa tiền tố nhân tạo '{c[:15]}...'.")
                    chunk_prefix_warnings += 1
                    errors += 1

            # 6. Kiểm tra nghiêm ngặt số lượng và vị trí token trạng thái
            reply = data.get("assistant_reasoning_and_reply", "")
            thought_cnt = reply.count("<|thought|>")
            close_thought_cnt = reply.count("<|/thought|>")
            answer_cnt = reply.count("<|answer|>")
            end_cnt = reply.count("<|endoftext|>")

            if (
                thought_cnt == 1
                and close_thought_cnt == 1
                and answer_cnt == 1
                and end_cnt == 1
                and STRICT_FORMAT_REGEX.match(reply)
            ):
                valid_tags_count += 1
            else:
                print(
                    f"⚠️ Dòng {idx} ({data.get('id')}): Vi phạm cấu trúc token! "
                    f"(thought={thought_cnt}, /thought={close_thought_cnt}, answer={answer_cnt}, endoftext={end_cnt})"
                )
                errors += 1

            # Xáo trộn chunks nếu có cờ --shuffle
            if shuffle_chunks:
                random.shuffle(data["external_memory_chunks"])

            records.append(data)

    # Ghi lại file khi shuffle và không có lỗi cấu trúc cơ bản
    if shuffle_chunks and errors == 0:
        with open(file_path, "w", encoding="utf-8") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n🎲 Đã xáo trộn ngẫu nhiên external_memory_chunks cho toàn bộ {len(records)} mẫu trong: {file_path}")

    print("-" * 60)
    print(f"📊 Kết quả kiểm tra Dataset:")
    print(f"  • Tổng số mẫu: {total}")
    print(f"  • Số mẫu đạt chuẩn token (xuất hiện đúng 1 lần): {valid_tags_count}/{total}")
    print(f"  • Trung bình số chunk/mẫu: {total_chunks / total if total > 0 else 0:.1f}")
    print(f"  • Phân bổ Domain:")
    for d, cnt in domains.most_common():
        print(f"      - {d}: {cnt} ({cnt / total * 100:.1f}%)")
    print(f"  • Phân bổ Category:")
    for c, cnt in categories.most_common():
        print(f"      - {c}: {cnt} ({cnt / total * 100:.1f}%)")
    print(f"  • Cảnh báo tiền tố nhân tạo: {chunk_prefix_warnings}")
    print(f"  • Tổng số lỗi phát hiện: {errors}")

    if errors == 0:
        print("\n✅ Toàn bộ mẫu dữ liệu ĐẠT CHUẨN (Passed)! 🚀")
        return 0
    else:
        print(f"\n❌ Phát hiện {errors} lỗi cần xử lý.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra cấu trúc và xáo trộn vị trí chunks cho dataset suy luận.")
    parser.add_argument("--file", type=str, default="data/reasoning_dataset.jsonl", help="Đường dẫn tới file .jsonl")
    parser.add_argument("--shuffle", action="store_true", help="Xáo trộn ngẫu nhiên vị trí external_memory_chunks để chống thiên kiến vị trí (Positional Bias)")
    args = parser.parse_args()
    code = validate_and_process(args.file, args.shuffle)
    sys.exit(code)

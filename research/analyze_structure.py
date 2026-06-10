import os
import random
import re
from pathlib import Path

def analyze_markdown(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        return

    print(f"Analyzing {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    total_lines = len(lines)
    print(f"Total lines: {total_lines}")
    print(f"Approximate character count: {len(content)}")

    # 1. Heading analysis
    headings = []
    source_markers = []
    table_lines = 0
    image_markers = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# --- Source:") and stripped.endswith("---"):
            source_markers.append((i, stripped))
        elif stripped.startswith("#"):
            # Check heading level
            match = re.match(r"^(#+)\s+(.*)$", stripped)
            if match:
                level = len(match.group(1))
                text = match.group(2)
                headings.append((i, level, text))
        elif stripped.startswith("|") and stripped.endswith("|"):
            table_lines += 1
        elif "<!-- image -->" in stripped:
            image_markers += 1

    print(f"\n--- General Content Stats ---")
    print(f"Source markers found: {len(source_markers)}")
    print(f"Total headings found: {len(headings)}")
    print(f"Total table lines: {table_lines}")
    print(f"Total image markers: {image_markers}")

    # Heading level distributions
    level_counts = {}
    for _, lvl, _ in headings:
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    print(f"Heading Level Distribution: {level_counts}")

    # Source markers sample
    print(f"\n--- Source Markers (all): ---")
    for idx, marker in source_markers:
        print(f"  Line {idx}: {marker}")

    # Heading sample
    print(f"\n--- Random Headings Sample (20 headings): ---")
    if headings:
        sampled_headings = random.sample(headings, min(len(headings), 20))
        # Sort them by line number
        sampled_headings.sort(key=lambda x: x[0])
        for idx, lvl, text in sampled_headings:
            print(f"  Line {idx} (L{lvl}): {text}")

    # Sample random blocks of text (including potential special boxes)
    print(f"\n--- Random 5-Line Blocks (3 samples): ---")
    for s in range(3):
        start = random.randint(0, max(0, total_lines - 10))
        print(f"\n[Sample {s+1} - Line {start} to {start+5}]:")
        for idx in range(start, min(total_lines, start + 5)):
            print(f"  {idx}: {lines[idx]}")

    # Search for special medical box patterns (e.g., "Emergency", "Practice Point", "In Old Age", "In Pregnancy")
    box_patterns = [
        r"Emergency",
        r"Practice Point",
        r"In Old Age",
        r"In Pregnancy",
        r"In Adolescence"
    ]
    
    print(f"\n--- Searching for special Box keywords/patterns (sample lines): ---")
    for pattern in box_patterns:
        matches = []
        for i, line in enumerate(lines):
            if re.search(r"\b" + pattern + r"\b", line, re.IGNORECASE):
                # Ensure it's in a header or distinct line
                if line.strip().startswith("##") or line.strip().startswith("|") or len(line.strip()) < 50:
                    matches.append((i, line.strip()))
        print(f"Pattern '{pattern}': {len(matches)} matches found.")
        if matches:
            sampled_matches = random.sample(matches, min(len(matches), 3))
            for idx, text in sampled_matches:
                print(f"  Line {idx}: {text}")

if __name__ == "__main__":
    analyze_markdown("Data/Output/medical_data.md")

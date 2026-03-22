#!/usr/bin/env python3
"""
MOM Session Transformer - Diff Analyzer

Compares the original .jsonl session file with the transformed JSON
to report what information was lost/retained in the transformation.

Usage:
    uv run python mom_session_diff.py [original_jsonl] [transformed_json]
"""

import json
import sys
from pathlib import Path
from typing import Any


def count_tokens_from_usage(messages: list) -> dict:
    """Sum tokens from message usage fields."""
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_tokens = 0

    for msg in messages:
        usage = msg.get("message", {}).get("usage", {})
        total_input += usage.get("input", 0)
        total_output += usage.get("output", 0)
        total_cache_read += usage.get("cacheRead", 0)
        total_cache_write += usage.get("cacheWrite", 0)
        total_tokens += usage.get("totalTokens", 0)

    return {
        "input": total_input,
        "output": total_output,
        "cache_read": total_cache_read,
        "cache_write": total_cache_write,
        "total": total_tokens,
    }


def count_content_blocks(messages: list) -> dict:
    """Count content block types in messages."""
    counts = {
        "text": 0,
        "toolCall": 0,
        "toolResult": 0,
        "citation": 0,
        "other": 0,
    }

    for msg in messages:
        content = msg.get("message", {}).get("content", [])
        for block in content:
            block_type = block.get("type", "unknown")
            if block_type in counts:
                counts[block_type] += 1
            else:
                counts["other"] += 1

    return counts


def count_tool_calls(messages: list) -> int:
    """Count total tool calls in messages."""
    count = 0
    for msg in messages:
        content = msg.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "toolCall":
                count += 1
    return count


def extract_text_content(messages: list) -> int:
    """Count total text characters in messages."""
    total = 0
    for msg in messages:
        content = msg.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "text":
                total += len(block.get("text", ""))
    return total


def analyze_original(jsonl_path: Path) -> dict:
    """Analyze original JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    messages = []
    for line in lines:
        d = json.loads(line)
        if d.get("type") == "message":
            messages.append(d)

    return {
        "total_lines": len(lines),
        "total_messages": len(messages),
        "user_messages": sum(1 for m in messages if m.get("message", {}).get("role") == "user"),
        "assistant_messages": sum(1 for m in messages if m.get("message", {}).get("role") == "assistant"),
        "tokens": count_tokens_from_usage(messages),
        "content_blocks": count_content_blocks(messages),
        "tool_calls": count_tool_calls(messages),
        "text_characters": extract_text_content(messages),
    }


def analyze_transformed(json_path: Path) -> dict:
    """Analyze transformed JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    turns = data.get("turns", [])

    # Count tool calls and text from turns
    tool_calls = 0
    text_chars = 0
    citations = 0

    for turn in turns:
        resp = turn.get("assistant_response", {})
        tool_calls += len(resp.get("tool_calls", []))
        text_chars += len(resp.get("final_output", ""))
        citations += len(resp.get("citations", []))

    return {
        "total_turns": len(turns),
        "total_tokens": data.get("total_tokens", 0),
        "tool_calls": tool_calls,
        "text_characters": text_chars,
        "citations": citations,
        "session_id": data.get("session_id", ""),
    }


def print_report(original: dict, transformed: dict):
    """Print comparison report."""
    print("=" * 60)
    print("MOM SESSION TRANSFORM - INFORMATION LOSS ANALYSIS")
    print("=" * 60)

    print("\n[SESSION STRUCTURE]")
    print(f"  Original: {original['total_lines']} lines (incl. session header + messages)")
    print(f"  Original: {original['total_messages']} messages")
    print(f"  Original: {original['user_messages']} user messages, {original['assistant_messages']} assistant messages")
    print(f"  Transformed: {transformed['total_turns']} turns (user-assistant pairs)")

    print("\n[MESSAGES]")
    print(f"  User messages kept: {transformed['total_turns']}/{original['user_messages']} turns")
    print(f"  Assistant responses kept: {transformed['total_turns']}/{original['assistant_messages']} turns")

    print("\n[TOKENS]")
    orig_tok = original["tokens"]
    print(f"  Original input tokens:  {orig_tok['input']:,}")
    print(f"  Original output tokens: {orig_tok['output']:,}")
    print(f"  Original cache read:    {orig_tok['cache_read']:,}")
    print(f"  Original cache write:  {orig_tok['cache_write']:,}")
    print(f"  Original total tokens: {orig_tok['total']:,}")
    print(f"  Transformed total:     {transformed['total_tokens']:,}")
    # Note: transformed only counts input+output from paired turns

    print("\n[CONTENT BLOCKS - Original]")
    blocks = original["content_blocks"]
    print(f"  text blocks:       {blocks['text']:>8,}")
    print(f"  toolCall blocks:   {blocks['toolCall']:>8,}")
    print(f"  toolResult blocks: {blocks['toolResult']:>8,}")
    print(f"  citation blocks:   {blocks['citation']:>8,}")
    print(f"  other blocks:      {blocks['other']:>8,}")

    print("\n[TOOL CALLS]")
    print(f"  Original: {original['tool_calls']} tool calls in content blocks")
    print(f"  Transformed: {transformed['tool_calls']} tool calls captured")
    print(f"  Retention: {transformed['tool_calls']}/{original['tool_calls']} = {transformed['tool_calls']*100/max(original['tool_calls'],1):.1f}%")

    print("\n[TEXT CONTENT]")
    print(f"  Original text characters: {original['text_characters']:,}")
    print(f"  Transformed text chars:  {transformed['text_characters']:,}")
    print(f"  Note: Original includes BOTH user queries AND assistant outputs")
    print(f"  Note: Transformed only captures assistant final_output")

    print("\n[CITATIONS]")
    print(f"  Transformed citations: {transformed['citations']}")

    print("\n[LOST INFORMATION]")
    print("  - session/cwd metadata: not in transformed")
    print("  - message.id, parentId: not in transformed")
    print("  - tool call outputs/results: not captured (only input preserved)")
    print("  - toolResult blocks: discarded")
    print("  - cache tokens: not counted in transformed")
    print("  - message-level model, api, provider: not in transformed")
    print("  - error messages (aborted, etc.): not in transformed")
    print("  - latency_ms: not available in original to compare")
    print("  - annotations layer: intentionally left empty (for external filling)")

    print("\n" + "=" * 60)


def main():
    if len(sys.argv) >= 3:
        original_path = Path(sys.argv[1])
        transformed_path = Path(sys.argv[2])
    else:
        base = Path("/Users/zhuanzmima0000/.mom/oc_084fbf2bc133a248e51217ce04276f1a/sessions")
        output_dir = Path("/Users/zhuanzmima0000/aeon/my_ai_brain/transform_output")

        # Find latest session
        jsonl_files = list(base.glob("*.jsonl"))
        latest_jsonl = sorted(jsonl_files, key=lambda f: f.name, reverse=True)[0]

        # Find matching transformed file
        transformed_files = list(output_dir.glob("*.json"))
        transformed_path = sorted(transformed_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]

        original_path = latest_jsonl

    print(f"Original:    {original_path}")
    print(f"Transformed: {transformed_path}")
    print()

    orig = analyze_original(original_path)
    trans = analyze_transformed(transformed_path)

    print_report(orig, trans)


if __name__ == "__main__":
    main()

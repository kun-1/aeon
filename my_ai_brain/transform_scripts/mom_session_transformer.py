#!/usr/bin/env python3
"""
MOM Session to Structured JSON Transformer

Transforms MOM session .jsonl files into a 3-tier structured JSON format:
- Session level: session metadata
- Turn level: user query + assistant response
- Annotation level: (empty, to be filled by external script)

Usage:
    python mom_session_transformer.py [input_dir] [output_dir]
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def find_latest_session_dir(sessions_dir: Path) -> Optional[Path]:
    """Find the latest session .jsonl file by timestamp in filename."""
    if not sessions_dir.exists():
        return None

    jsonl_files = list(sessions_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Sort by filename (contains ISO timestamp)
    latest = sorted(jsonl_files, key=lambda f: f.name, reverse=True)[0]
    return latest


def parse_timestamp(ts_str: str) -> str:
    """Convert ISO timestamp to desired format with timezone (+08:00)."""
    try:
        # Handle format: 2026-03-19T08-42-34-258Z
        if "-42-" in ts_str:
            dt = datetime.strptime(ts_str.replace("-42-", ":42:").replace("-", "")[:23],
                                  "%Y-%m-%dT%H:%M:%S.%f")
            return dt.isoformat() + "+08:00"
        # Handle format: 2026-03-19T08:42:34.258Z
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.isoformat().replace("+00:00", "+08:00")
    except Exception:
        return ts_str


def extract_messages(lines: list) -> dict:
    """Extract session metadata and messages from raw jsonl lines."""
    session_meta = {}
    messages = []

    for line in lines:
        d = json.loads(line)
        t = d.get("type")

        if t == "session":
            session_meta = {
                "session_id": d.get("id", "")[:20],
                "start_time": parse_timestamp(d.get("timestamp", "")),
                "cwd": d.get("cwd", ""),
            }
        elif t == "tool_execution_start":
            messages.append({
                "id": d.get("id", ""),
                "parent_id": d.get("parentId"),
                "timestamp": parse_timestamp(d.get("timestamp", "")),
                "role": "tool_execution_start",
                "tool_call_id": d.get("toolCallId", ""),
                "tool_name": d.get("toolName", ""),
                "args": d.get("args", {}),
                "retry_count": d.get("retryCount", 0),
            })
        elif t == "tool_execution_end":
            result = d.get("result", {})
            content = result.get("content", [])
            # Extract text from content
            result_text = ""
            if isinstance(content, list):
                texts = [block.get("text", "") for block in content if block.get("type") == "text"]
                result_text = "\n".join(texts)

            messages.append({
                "id": d.get("id", ""),
                "parent_id": d.get("parentId"),
                "timestamp": parse_timestamp(d.get("timestamp", "")),
                "role": "tool_execution_end",
                "tool_call_id": d.get("toolCallId", ""),
                "tool_name": d.get("toolName", ""),
                "is_error": d.get("isError", False),
                "duration_ms": d.get("durationMs", 0),
                "result": result_text,
            })
        elif t == "message":
            msg = d.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", [])

            # For toolResult, extract tool_call_id to link with original call
            tool_call_id = None
            if role == "toolResult":
                tool_call_id = msg.get("toolCallId", msg.get("tool_call_id"))

            messages.append({
                "id": d.get("id", ""),
                "parent_id": d.get("parentId"),
                "timestamp": parse_timestamp(d.get("timestamp", "")),
                "role": role,
                "content": content,
                "tool_call_id": tool_call_id,
                "model": msg.get("model", ""),
                "usage": msg.get("usage", {}),
                "stop_reason": msg.get("stopReason", ""),
            })

    return {
        "session_meta": session_meta,
        "messages": messages,
    }


def _extract_text_from_content(content: list) -> str:
    """Extract text from content blocks."""
    if isinstance(content, list):
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)
    return str(content) if content else ""


def _finalize_turn(current_turn: dict, assistant_msgs: list, tool_events: dict) -> dict:
    """Helper to finalize a turn with accumulated assistant messages and tool events."""
    if current_turn is None:
        return None

    all_tool_calls = []
    final_output = ""
    citations = []

    # Process tool events (from tool_execution_start/end)
    for tc_id, event in tool_events.items():
        start = event.get("start", {})
        end = event.get("end", {})
        result = end.get("result", "") if end else ""

        # Truncate args.content to 100 chars only
        args = start.get("args", {})
        if "content" in args and isinstance(args["content"], str) and len(args["content"]) > 100:
            args["content"] = args["content"][:100]

        all_tool_calls.append({
            "tool_call_id": tc_id,
            "tool_name": start.get("tool_name", ""),
            "args": args,
            "retry_count": start.get("retry_count", 0),
            "is_error": end.get("is_error", False) if end else False,
            "duration_ms": end.get("duration_ms", 0) if end else 0,
            "result": result,
        })

    for asst_msg in assistant_msgs:
        for block in asst_msg.get("content", []):
            if block.get("type") == "text":
                final_output = block.get("text", "")
            elif block.get("type") == "citation":
                citations.append(f"web:{block.get('index', '?')}")

        # Accumulate tokens
        usage = asst_msg.get("usage", {})
        if "tokens_used" in current_turn:
            current_turn["tokens_used"]["input"] += usage.get("input", 0)
            current_turn["tokens_used"]["output"] += usage.get("output", 0)

    current_turn["assistant_response"]["tool_calls"] = all_tool_calls
    current_turn["assistant_response"]["final_output"] = final_output
    current_turn["assistant_response"]["citations"] = citations

    return current_turn


def build_turns(messages: list) -> list:
    """
    Pair user queries with assistant responses to form turns.

    A turn consists of:
    - One user message (with text query)
    - One or more assistant messages
    - tool_execution_start/end events

    The turn is complete when:
    - We encounter a new user message, OR
    - The assistant produces a final text output (no more tool calls expected)
    """
    turns = []
    current_turn = None
    current_assistant_msgs = []
    current_tool_events = {}  # tool_call_id -> {start, end}

    for msg in messages:
        role = msg["role"]

        if role == "tool_execution_start":
            # Store tool execution start event
            tc_id = msg.get("tool_call_id", "")
            current_tool_events[tc_id] = {
                "start": {
                    "tool_call_id": tc_id,
                    "tool_name": msg.get("tool_name", ""),
                    "args": msg.get("args", {}),
                    "retry_count": msg.get("retry_count", 0),
                    "timestamp": msg.get("timestamp", ""),
                },
                "end": None
            }

        elif role == "tool_execution_end":
            # Store tool execution end event
            tc_id = msg.get("tool_call_id", "")
            if tc_id in current_tool_events:
                current_tool_events[tc_id]["end"] = {
                    "tool_call_id": tc_id,
                    "tool_name": msg.get("tool_name", ""),
                    "is_error": msg.get("is_error", False),
                    "duration_ms": msg.get("duration_ms", 0),
                    "result": msg.get("result", ""),
                    "timestamp": msg.get("timestamp", ""),
                }

        elif role == "user":
            # Finalize previous turn if exists
            if current_turn is not None and current_assistant_msgs:
                finalized = _finalize_turn(current_turn, current_assistant_msgs, current_tool_events)
                if finalized:
                    turns.append(finalized)
                current_tool_events = {}

            # Start new turn with user query
            query = ""
            for block in msg.get("content", []):
                if block.get("type") == "text":
                    query = block.get("text", "")
                    break

            usage = msg.get("usage", {})
            current_turn = {
                "session_id": "",
                "timestamp": msg["timestamp"],
                "speaker": "user",
                "query": query,
                "turn_type": "",
                "has_memory_signal": None,
                "assistant_response": {
                    "think": "",
                    "tool_calls": [],
                    "final_output": "",
                    "citations": [],
                },
                "latency_ms": 0,
                "tokens_used": {
                    "input": usage.get("input", 0),
                    "output": usage.get("output", 0),
                },
                "annotations": {
                    "memory_candidates": [],
                    "need_category": "",
                    "is_correction": False,
                    "user_feedback_signal": "",
                },
            }
            current_assistant_msgs = []

        elif role == "assistant":
            # Check if this assistant has a final text output
            has_final_text = any(
                block.get("type") == "text" and block.get("text", "").strip()
                for block in msg.get("content", [])
            )

            # If current_turn is None but we have accumulated messages,
            # create a new turn to hold them
            if current_turn is None and current_assistant_msgs:
                current_turn = {
                    "session_id": "",
                    "timestamp": "",
                    "speaker": "user",
                    "query": "[continuation]",
                    "turn_type": "",
                    "has_memory_signal": None,
                    "assistant_response": {
                        "think": "",
                        "tool_calls": [],
                        "final_output": "",
                        "citations": [],
                    },
                    "latency_ms": 0,
                    "tokens_used": {
                        "input": 0,
                        "output": 0,
                    },
                    "annotations": {
                        "memory_candidates": [],
                        "need_category": "",
                        "is_correction": False,
                        "user_feedback_signal": "",
                    },
                }

            current_assistant_msgs.append(msg)

            # If assistant has final text output, end the turn immediately
            if has_final_text:
                finalized = _finalize_turn(current_turn, current_assistant_msgs, current_tool_events)
                if finalized:
                    turns.append(finalized)
                current_turn = None
                current_assistant_msgs = []
                current_tool_events = {}

    # Handle last turn if exists
    if current_turn is not None and current_assistant_msgs:
        finalized = _finalize_turn(current_turn, current_assistant_msgs, current_tool_events)
        if finalized:
            turns.append(finalized)

    return turns


def generate_session_id(start_time: str) -> str:
    """Generate session ID from start time."""
    # Format: s_20260319_001
    try:
        dt = datetime.fromisoformat(start_time.replace("+08:00", ""))
    except Exception:
        dt = datetime.fromisoformat(start_time.replace("+00:00", ""))
    date_str = dt.strftime("%Y%m%d")
    seq = dt.strftime("%H%M")[1:]  # Skip first digit to reduce uniqueness
    return f"s_{date_str}_{seq}"


def transform(input_path: Path, output_dir: Path) -> dict:
    """Main transformation function."""
    # Read raw jsonl
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse
    parsed = extract_messages(lines)
    session_meta = parsed["session_meta"]
    messages = parsed["messages"]

    # Build turns
    turns = build_turns(messages)

    # Generate session info
    start_time = session_meta.get("start_time", "")
    session_id = generate_session_id(start_time)

    # Update turns with session_id
    for turn in turns:
        turn["session_id"] = session_id

    # Final structured output
    result = {
        "session_id": session_id,
        "user_id": "",  # Not available
        "start_time": start_time,
        "end_time": turns[-1]["timestamp"] if turns else "",
        "session_topic": "",  # Not available
        "language": "zh-CN",  # Default assumption
        "model_used": messages[1]["model"] if len(messages) > 1 else "",  # assistant's model
        "total_turns": len(turns),
        "total_tokens": sum(
            t["tokens_used"]["input"] + t["tokens_used"]["output"]
            for t in turns
        ),
        "turns": turns,
    }

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{session_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {
        "session_id": session_id,
        "turns_count": len(turns),
        "output_file": str(output_file),
    }


def main():
    if len(sys.argv) >= 3:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    else:
        # Default paths
        base = Path("/Users/zhuanzmima0000/.mom/oc_084fbf2bc133a248e51217ce04276f1a")
        input_dir = base / "sessions"
        output_dir = Path("/Users/zhuanzmima0000/aeon/my_ai_brain/transform_output")

    latest_session = find_latest_session_dir(input_dir)
    if not latest_session:
        print(f"No session files found in {input_dir}")
        sys.exit(1)

    print(f"Processing: {latest_session.name}")
    result = transform(latest_session, output_dir)
    print(f"Output: {result['output_file']}")
    print(f"Session ID: {result['session_id']}")
    print(f"Turns: {result['turns_count']}")


if __name__ == "__main__":
    main()

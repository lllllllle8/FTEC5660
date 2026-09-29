#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once."""
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda
    from langchain_deepseek import ChatDeepSeek

    system_prompt = (
        "You are a meticulous transcriber of Hong Kong supermarket receipts. "
        "Read the receipt image line by line and output ONLY one JSON object, "
        "with no markdown fences and no commentary.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "store": string,\n'
        '  "items": [{"name": string, "amount": number}],\n'
        '  "discounts": [{"name": string, "amount": number}],\n'
        '  "subtotal": number or null,\n'
        '  "rounding": number,\n'
        '  "total_paid": number\n'
        "}\n\n"
        "Rules:\n"
        "1. items: every purchased product line, using the LINE TOTAL printed on the "
        "right (for '2 @ 5.00' lines use 10.00, not 5.00). Positive numbers only. "
        "Do not include SUBTOTAL, TOTAL, payment, change, or points lines.\n"
        "2. discounts: EVERY price reduction line, wherever it appears (under an item "
        "or near the total): promotion, coupon, member, app, % OFF, bundle offer, "
        "markdown, packaging-damage, cash voucher, etc. Record each as a POSITIVE "
        "number (e.g. '-5.39' becomes 5.39). Rounding is NOT a discount.\n"
        "3. subtotal: the value printed next to SUBTOTAL, or null if absent.\n"
        "4. rounding: the signed ROUNDING / ADJUSTMENT amount exactly as printed "
        "(e.g. -0.01 or 0.02). Use 0 if there is no rounding line.\n"
        "5. total_paid: the final amount the customer actually had to pay after "
        "rounding (the TOTAL / amount due / OCTOPUS / card charge). If paid by cash "
        "with change, use the amount due, NOT the cash tendered.\n"
        "6. Check yourself before answering: sum(items) - sum(discounts) + rounding "
        "should equal total_paid. If it does not, re-read the receipt.\n"
        "7. Output numbers as plain decimals without currency symbols or commas."
    )

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=3,
        timeout=180,
    )

    def to_messages(inputs: dict) -> list:
        text = "Transcribe this supermarket receipt into the JSON schema."
        if inputs.get("feedback"):
            text += "\n\nIMPORTANT FEEDBACK ON A PREVIOUS ATTEMPT:\n" + inputs["feedback"]
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": inputs["image_url"]}},
                ]
            ),
        ]

    # Chain: build multimodal messages -> vision LLM -> raw text (JSON parsed later)
    return RunnableLambda(to_messages) | llm | StrOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string."""
    ### YOUR CODE HERE
    from collections import Counter

    cent = Decimal("0.01")
    tolerance = Decimal("0.015")
    max_rounds = 3  # 1 extraction pass + up to 2 reflection retries

    def to_dec(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
        return Decimal(match.group()) if match else Decimal("0")

    def parse(text: Any) -> dict | None:
        """Parse the model's JSON and derive the two per-receipt amounts."""
        if not isinstance(text, str):
            return None
        text = re.sub(r"```(?:json)?", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None

        items_sum, disc_sum = Decimal("0"), Decimal("0")
        for row in data.get("items") or []:
            amount = to_dec(row.get("amount") if isinstance(row, dict) else row)
            if amount < 0:  # a negative "item" is really a discount line
                disc_sum += -amount
            else:
                items_sum += amount
        for row in data.get("discounts") or []:
            disc_sum += abs(to_dec(row.get("amount") if isinstance(row, dict) else row))

        rounding = to_dec(data.get("rounding"))
        paid = abs(to_dec(data.get("total_paid")))
        if paid == 0:
            return None

        consistent = abs(items_sum - disc_sum + rounding - paid) <= tolerance
        if not consistent and rounding != 0:
            # the model may have flipped the sign of the rounding line
            if abs(items_sum - disc_sum - rounding - paid) <= tolerance:
                rounding, consistent = -rounding, True

        return {
            "paid": paid.quantize(cent),
            # original price = what was paid, minus rounding, plus every discount
            "orig": (paid - rounding + disc_sum).quantize(cent),
            "items": items_sum,
            "disc": disc_sum,
            "rounding": rounding,
            "consistent": consistent,
        }

    def feedback_for(rec: dict | None) -> str:
        if rec is None:
            return (
                "Your previous answer was not a valid JSON object with a total_paid "
                "value. Output only the JSON object described in the instructions."
            )
        return (
            f"A previous reading was arithmetically inconsistent: items summed to "
            f"{rec['items']}, discounts summed to {rec['disc']}, rounding was "
            f"{rec['rounding']}, but total_paid was {rec['paid']}. "
            "sum(items) - sum(discounts) + rounding must equal total_paid. Re-read "
            "every line carefully (multi-quantity line totals, discount lines under "
            "items, the rounding line, and the final amount due) and output a "
            "corrected JSON object."
        )

    def pick(cands: list[dict]) -> dict | None:
        """Prefer self-consistent extractions; majority vote among them."""
        if not cands:
            return None
        pool = [c for c in cands if c["consistent"]] or cands
        votes = Counter((c["paid"], c["orig"]) for c in pool)
        best_key = votes.most_common(1)[0][0]
        return next(c for c in pool if (c["paid"], c["orig"]) == best_key)

    urls = [image_data_url(path) for path in images]
    candidates: list[list[dict]] = [[] for _ in images]
    feedback: list[str | None] = [None] * len(images)
    pending = list(range(len(images)))

    for _round in range(max_rounds):
        if not pending:
            break
        batch_inputs = [{"image_url": urls[i], "feedback": feedback[i]} for i in pending]
        try:
            outputs = chain.batch(
                batch_inputs, config={"max_concurrency": 8}, return_exceptions=True
            )
        except Exception as exc:  # never crash: keep whatever we already have
            print(f"batch call failed: {exc}")
            outputs = [exc] * len(pending)

        still_pending = []
        for i, out in zip(pending, outputs):
            if isinstance(out, Exception):
                print(f"{images[i].name}: model error: {out}")
                rec = None
            else:
                rec = parse(out)
            if rec is not None:
                candidates[i].append(rec)
            if rec is not None and rec["consistent"]:
                continue
            feedback[i] = feedback_for(rec)
            still_pending.append(i)
        pending = still_pending

    total_paid, total_orig = Decimal("0"), Decimal("0")
    for path, cands in zip(images, candidates):
        best = pick(cands)
        if best is None:
            print(f"{path.name}: no usable extraction")
            continue
        flag = "" if best["consistent"] else "  (inconsistent, best effort)"
        print(f"{path.name}: paid {best['paid']}, without discount {best['orig']}{flag}")
        total_paid += best["paid"]
        total_orig += best["orig"]

    return {
        QUERY_1: f"HK${total_paid.quantize(cent):.2f}",
        QUERY_2: f"HK${total_orig.quantize(cent):.2f}",
    }

# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

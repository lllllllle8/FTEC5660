# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 

```mermaid
flowchart TD
    A[Receipt folder] --> B[Encode each image as base64 data URL]
    B --> C[Parallel batch: prompt + image to deepseek-v4-flash-vision-exp]
    C --> D[Parse JSON: items, discounts, rounding, total_paid]
    D --> E{items - discounts + rounding == total_paid?}
    E -- yes --> G[Accept extraction]
    E -- no, up to 2 retries --> F[Reflection: send the mismatch back as feedback]
    F --> C
    G --> H[Python Decimal aggregation across receipts]
    H --> I[Q1: sum of total_paid]
    H --> J[Q2: sum of total_paid - rounding + discounts]
```

My chain separates perception from arithmetic. 
For every receipt, a LangChain runnable (message builder, then ChatDeepSeek with deepseek-v4-flash-vision-exp at temperature 0, then a string parser) transcribes the image into a structured JSON object containing every item line total, every discount line as a positive number, the signed rounding line, and the final amount paid. 
All receipts are processed in parallel with `batch`. Each extraction is validated with an arithmetic consistency，check (items minus discounts plus rounding must equal the final payment); when the check fails, the concrete mismatch is sent back to the model as feedback for up to two reflection retries, and a majority vote over consistent readings picks the final extraction. 
The totals are then computed deterministically in Python with `Decimal`:
Query 1 sums the final payments, and Query 2 sums each payment minus its rounding plus all discounts, which recovers the pre-discount price regardless of whether SUBTOTAL is printed before or after the discount lines. 
The response for each query contains exactly one HKD amount, for example `HK$1974.30`.


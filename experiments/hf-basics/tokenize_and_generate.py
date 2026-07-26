"""Session 6 — look under the Ollama server abstraction.

1) Tokenize a sentence -> print token IDs and decoded pieces
2) Run a tiny CPU text-generation example

Uses a small GPT-2 model so download/CPU run stays manageable.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_ID = "sshleifer/tiny-gpt2"
SENTENCE = "The dishwasher will not drain because"


def show_tokenization(tokenizer: AutoTokenizer, text: str) -> None:
    encoded = tokenizer(text)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print("=== Tokenization ===")
    print(f"text: {text!r}")
    print(f"token ids ({len(token_ids)}): {token_ids}")
    print("id -> token:")
    for token_id, token in zip(token_ids, tokens, strict=True):
        # GPT-2 marks word starts with 'Ġ'; escape for Windows consoles.
        safe_token = token.encode("unicode_escape").decode("ascii")
        print(f"  {token_id:>6} -> {safe_token}")
    print(f"decoded: {tokenizer.decode(token_ids)!r}")
    print()


def run_generation(model_id: str, prompt: str) -> None:
    print("=== Tiny text generation (CPU) ===")
    print(f"model: {model_id}")
    print(f"prompt: {prompt!r}")

    generator = pipeline(
        "text-generation",
        model=model_id,
        device=-1,  # CPU
    )
    outputs = generator(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
    )
    print("generated:")
    print(outputs[0]["generated_text"])
    print()


def main() -> None:
    print(f"Loading tokenizer/model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Touch the model once so weights are downloaded before generation.
    _ = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    show_tokenization(tokenizer, SENTENCE)
    run_generation(MODEL_ID, SENTENCE)


if __name__ == "__main__":
    main()

import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def extract_features(text: str, vocab: set) -> dict:
    cleaned = clean_text(text)
    words = cleaned.split()
    unknown = sum(1 for w in words if w not in vocab)
    return {
        "text_length": len(cleaned),
        "word_count": len(words),
        "unknown_ratio": round(unknown / len(words), 4) if words else 0.0,
    }


def extract_features_batch(texts: list, vocab: set) -> list:
    return [extract_features(t, vocab) for t in texts]

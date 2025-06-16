import re


def preprocess(text: str) -> str:
    text = re.sub(r"@user", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return text.strip()

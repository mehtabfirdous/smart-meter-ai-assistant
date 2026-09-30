import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text without removing
    important technical information.
    """

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


if __name__ == "__main__":

    sample_text = """
    Smart Meter     Communication


    Meter data is commonly transmitted
    through a communication network to HES.
    """

    cleaned = clean_text(sample_text)

    print("Before:")
    print(sample_text)

    print("\nAfter:")
    print(cleaned)
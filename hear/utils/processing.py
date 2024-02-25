import re


def word_tokenize(text, exclude_highlight=False):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    if not exclude_highlight:
        tokenized_text = [
            s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(text.strip())
            if len(s.strip()) > 0]
    else:
        tokenized_text = [
            s.strip().lower() for s in re.findall(r'\[[^\]]*\]|[\w]+|[.,!?;]', text)
            if len(s.strip()) > 0]

    return tokenized_text


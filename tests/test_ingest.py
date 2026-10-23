from paperlens.ingest import clean, split_text


def test_clean_joins_hyphenated_words():
    assert clean("classi-\nfication works") == "classification works"


def test_split_text_respects_size_and_keeps_content():
    text = " ".join(f"Sentence number {i} is about neural networks." for i in range(200))
    chunks = split_text(text, size=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 80 for c in chunks)
    assert "Sentence number 199" in chunks[-1]


def test_short_fragments_are_dropped():
    assert split_text("Fig. 1") == []

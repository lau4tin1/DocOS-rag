from docrag.retrieve.bm25 import BM25Index, tokenize


def test_tokenize_english_code_chinese():
    assert tokenize("Hello world") == ["hello", "world"]
    assert tokenize("ERR_SOCKET_502") == ["err_socket_502"]
    assert tokenize("你好") == ["你", "好"]


def test_unique_term_ranks_first():
    corpus = [
        "the quick brown fox",
        "the quick brown fox jumps over the lazy dog",
        "completely unrelated words here",
    ]
    bm = BM25Index()
    bm.fit(corpus)
    ranked = bm.search("unrelated")
    assert ranked[0][1] == 2  # "unrelated" 只出现在第 3 篇


def test_rarer_term_has_higher_idf():
    corpus = [
        "the quick brown fox",
        "the quick brown fox jumps",
        "completely unrelated words",
    ]
    bm = BM25Index()
    bm.fit(corpus)
    # quick 出现在 2 篇,unrelated 只出现在 1 篇 -> 后者 idf 更大
    assert bm._idf("unrelated") > bm._idf("quick")

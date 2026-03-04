from funding_extraction.thinking import strip_thinking


def test_strip_closed_think_tags():
    text = '<think>some reasoning</think>{"result": true}'
    content, traces = strip_thinking(text)
    assert content == '{"result": true}'
    assert traces == ["some reasoning"]


def test_strip_multiple_think_tags():
    text = '<think>first</think>hello<think>second</think>world'
    content, traces = strip_thinking(text)
    assert content == "helloworld"
    assert traces == ["first", "second"]


def test_strip_unclosed_think_tag():
    text = '<think>truncated reasoning here'
    content, traces = strip_thinking(text)
    assert content == ""
    assert traces == ["truncated reasoning here"]


def test_no_think_tags():
    text = '{"funders": []}'
    content, traces = strip_thinking(text)
    assert content == '{"funders": []}'
    assert traces == []


def test_empty_think_tags():
    text = '<think></think>{"result": true}'
    content, traces = strip_thinking(text)
    assert content == '{"result": true}'
    assert traces == []


def test_multiline_think_content():
    text = '<think>\nline1\nline2\n</think>\n[{"funder_name": "NSF"}]'
    content, traces = strip_thinking(text)
    assert "[" in content
    assert traces == ["line1\nline2"]


def test_think_tag_with_whitespace_only():
    text = '<think>   \n  </think>output'
    content, traces = strip_thinking(text)
    assert content == "output"
    assert traces == []

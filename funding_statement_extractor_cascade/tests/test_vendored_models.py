def test_chunk_classifier_class_is_importable():
    from funding_statement_extractor_cascade.models.chunk_classifier.modeling import ChunkClassifier
    assert ChunkClassifier.__name__ == "ChunkClassifier"


def test_span_head_class_is_importable():
    from funding_statement_extractor_cascade.models.span_head.modeling import SpanHead
    assert SpanHead.__name__ == "SpanHead"

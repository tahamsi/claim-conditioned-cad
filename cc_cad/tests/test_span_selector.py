from cc_cad.entailment.span_select import select_best_span


def test_select_best_span() -> None:
    passage = "Paris is the capital of France and a major European city." 
    claim = "Paris is the capital of France."
    span = select_best_span(passage, claim, window_size=8)
    assert span.end > span.start
    assert passage[span.start:span.end].lower().find("paris") >= 0

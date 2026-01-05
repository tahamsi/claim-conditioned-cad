from cc_cad.entailment.nli import NLIScorer


def test_entailment_mock() -> None:
    scorer = NLIScorer(model_name="mock")
    score = scorer.score_entailment("Paris is in France", "Paris is in France")
    assert score >= 0.9

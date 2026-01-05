from cc_cad.eval.claim_split import split_claims


def test_split_claims_basic() -> None:
    text = "Paris is in France. It is a capital city; it is large." 
    claims = split_claims(text)
    assert len(claims) == 3
    assert claims[0].startswith("Paris")


def test_split_claims_abbrev() -> None:
    text = "Dr. Smith lives in Boston. He works at MGH." 
    claims = split_claims(text)
    assert len(claims) == 2

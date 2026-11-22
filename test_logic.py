from crag_logic import decide_confidence, split_into_strips, filter_strips, recompose

def test_confidence():
    assert decide_confidence([0.9, 0.1], upper=0.6, lower=0.3) == "correct"
    assert decide_confidence([0.1, 0.2], upper=0.6, lower=0.3) == "incorrect"
    assert decide_confidence([0.4, 0.5], upper=0.6, lower=0.3) == "ambiguous"
    assert decide_confidence([0.1, 0.4], upper=0.6, lower=0.3) == "ambiguous"
    assert decide_confidence([], upper=0.6, lower=0.3) == "incorrect"
    assert decide_confidence([0.6], upper=0.6, lower=0.3) == "ambiguous"  # not strictly above

def test_split_and_filter():
    text = "One. Two! Three? Four. Five."
    strips = split_into_strips(text, 2)
    assert strips == ["One. Two!", "Three? Four.", "Five."]
    kept = filter_strips(strips, [0.9, 0.1, 0.7], threshold=0.5)
    assert kept == ["One. Two!", "Five."]
    assert recompose(kept) == "One. Two!\nFive."
    assert filter_strips(strips, [0.9, 0.8, 0.7], 0.5, top_k=2) == ["One. Two!", "Three? Four."]
    assert filter_strips(strips, [0.1, 0.1, 0.1], 0.5) == []

if __name__ == "__main__":
    test_confidence(); test_split_and_filter(); print("all logic tests passed")

def test_parse_and_sigmoid():
    from crag_logic import parse_score, sigmoid
    assert parse_score("0.8") == 0.8
    assert parse_score("Score: 0.35 because...") == 0.35
    assert parse_score("1") == 1.0
    assert parse_score("7") == 1.0          # clamped
    assert parse_score("no idea") == 0.0
    assert parse_score("") == 0.0
    assert abs(sigmoid(0) - 0.5) < 1e-9 and sigmoid(10) > 0.99 and sigmoid(-10) < 0.01

if __name__ == "__main__":
    test_parse_and_sigmoid(); print("parse/sigmoid tests passed")

"""Offline tests for the paper-probe orchestration layer."""

from pathlib import Path

import numpy as np
import pytest

from era.paper_probe import (
    EncodedPaperProbe,
    ModelObservations,
    PaperProbeSpec,
    encode_paper_probe,
    evaluate_observations,
    load_paper_probe_spec,
    measure_model,
    stable_softmax,
)

ROOT = Path(__file__).resolve().parent.parent


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(self.mapping[text])}


def _small_spec():
    return PaperProbeSpec(
        name="small",
        target_groups={"male": (" man",), "female": (" woman",)},
        concept_tokens=(" leader", " nurse"),
        top_k=2,
        top_p=0.8,
        log_base=None,
        source_sha256="abc",
    )


def test_repository_probe_is_fixed_14_targets_and_13_common_concepts():
    spec = load_paper_probe_spec(ROOT / "data" / "paper_probe_v1.json")
    assert sum(len(group) for group in spec.target_groups.values()) == 14
    assert len(spec.concept_tokens) == 13
    assert len(spec.source_sha256) == 64
    assert spec.top_k == 50
    assert spec.top_p == pytest.approx(0.95)


def test_probe_encoding_is_exact_and_never_silently_skips():
    tokenizer = FakeTokenizer(
        {
            " man": (1,),
            " woman": (2,),
            " leader": (3,),
            " nurse": (4,),
        }
    )
    encoded = encode_paper_probe(tokenizer, _small_spec())
    assert encoded.target_groups == {"male": (1,), "female": (2,)}
    assert encoded.concept_ids == (3, 4)

    tokenizer.mapping[" nurse"] = (4, 5)
    with pytest.raises(ValueError, match="maps to 2 tokens"):
        encode_paper_probe(tokenizer, _small_spec())


def test_probe_encoding_rejects_token_id_collisions():
    tokenizer = FakeTokenizer(
        {
            " man": (1,),
            " woman": (1,),
            " leader": (3,),
            " nurse": (4,),
        }
    )
    with pytest.raises(ValueError, match="overlapping token IDs"):
        encode_paper_probe(tokenizer, _small_spec())


def test_stable_softmax_matches_hand_value_without_smoothing():
    result = stable_softmax((0.0, 0.0, -1000.0))
    assert result[:2] == pytest.approx((0.5, 0.5))
    assert result[2] == 0.0  # representational underflow is recorded, not lifted
    assert result.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        stable_softmax((0.0, float("nan")))


def test_evaluate_observations_returns_all_paper_views_and_invariants():
    spec = _small_spec()
    encoded = EncodedPaperProbe(
        target_groups={"male": (0,), "female": (1,)},
        concept_ids=(2, 3),
        token_id_by_text={" man": 0, " woman": 1, " leader": 2, " nurse": 3},
    )
    p = np.array(
        [
            [0.4, 0.1, 0.2, 0.2, 0.1],
            [0.1, 0.4, 0.2, 0.2, 0.1],
        ]
    )
    q = np.array(
        [
            [0.5, 0.1, 0.15, 0.15, 0.1],
            [0.1, 0.5, 0.15, 0.15, 0.1],
        ]
    )
    one_context_states = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    states = np.stack((one_context_states, one_context_states))
    result = evaluate_observations(
        ModelObservations(p, states),
        ModelObservations(q, states),
        contexts=("lead", "support"),
        families=("leadership", "support"),
        spec=spec,
        encoded=encoded,
    )

    assert set(result["aggregates"]) == {"B", "B_k", "B_alpha", "B_T"}
    bt = result["aggregates"]["B_T"]
    assert bt["total"]["mean"] == pytest.approx(
        bt["between"]["mean"] + bt["within"]["mean"]
    )
    assert bt["within"]["mean"] == pytest.approx(0.0, abs=1e-15)
    assert result["SI"]["conditional"]["m0"]["SI"] == pytest.approx(1.2)
    assert result["SI"]["conditional"]["m1"]["SI"] == pytest.approx(4 / 3)
    assert result["SI"]["raw_probability_mass_diagnostic"]["delta_SI"] == (
        pytest.approx(0.2)
    )
    assert result["G_l"]["similarity"] == pytest.approx((1.0, 1.0))
    assert result["G_l"]["normalised_depth_centroid_of_drift"] is None
    assert len(result["per_context"]) == 2


def test_evaluate_observations_rejects_zero_target_mass():
    spec = _small_spec()
    encoded = EncodedPaperProbe(
        target_groups={"male": (0,), "female": (1,)},
        concept_ids=(2, 3),
        token_id_by_text={},
    )
    probabilities = np.array([[0.0, 0.0, 0.5, 0.5]])
    states = np.ones((1, 2, 1, 2))
    with pytest.raises(ValueError, match="target set T"):
        evaluate_observations(
            ModelObservations(probabilities, states),
            ModelObservations(probabilities, states),
            contexts=("context",),
            families=("leadership",),
            spec=spec,
            encoded=encoded,
        )


def test_measure_model_batches_concepts_at_the_exact_last_position():
    torch = pytest.importorskip("torch")

    class FakeEmbeddings:
        num_embeddings = 5

    class FakeOutput:
        pass

    class FakeModel:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return FakeEmbeddings()

        def __call__(self, input_ids, output_hidden_states=False, use_cache=None):
            output = FakeOutput()
            batch, sequence = input_ids.shape
            output.logits = torch.zeros((batch, sequence, 5), dtype=torch.float32)
            if output_hidden_states:
                token_values = input_ids.to(torch.float32).unsqueeze(-1)
                output.hidden_states = (
                    torch.cat((token_values, token_values + 1), dim=-1),
                    torch.cat((token_values + 2, token_values + 3), dim=-1),
                )
            return output

    tokenizer = FakeTokenizer({"ctx": (0, 1)})
    observed = measure_model(
        FakeModel(), tokenizer, ("ctx",), concept_ids=(2, 4), device="cpu"
    )
    assert observed.probabilities.shape == (1, 5)
    assert observed.probabilities[0] == pytest.approx(np.full(5, 0.2))
    assert observed.states.shape == (1, 2, 2, 2)
    # The two rows come from candidate IDs 2 and 4, proving that the measured
    # state is the appended final token, not the last token of the context.
    assert observed.states[0, :, 0, 0] == pytest.approx((2.0, 4.0))

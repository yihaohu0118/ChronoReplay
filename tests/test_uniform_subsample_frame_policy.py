from benchmark.ovo.utils.frame_utils import uniform_subsample_fixed_indices


def test_uniform_subsample_is_exact_subset_with_fixed_budget() -> None:
    end_frame = 10_000
    source = uniform_subsample_fixed_indices(
        end_frame, source_nframes=128, target_nframes=128
    )
    target = uniform_subsample_fixed_indices(
        end_frame, source_nframes=128, target_nframes=81
    )
    assert len(source) == 128
    assert len(target) == 81
    assert set(target) <= set(source)
    assert target[0] == source[0] == 0
    assert target[-1] == source[-1] == end_frame


def test_uniform_subsample_short_prefix_deduplicates() -> None:
    target = uniform_subsample_fixed_indices(
        20, source_nframes=128, target_nframes=81
    )
    assert target == list(range(21))

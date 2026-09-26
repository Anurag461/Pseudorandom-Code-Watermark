KGW_GAMMA, KGW_DELTA = (0.25, 2.0)


def _cpu_kgw_processor(vocab):
    import torch
    from watermarking.kirchenbauer.watermark_processor import WatermarkLogitsProcessor

    class CPUSeededKGW(WatermarkLogitsProcessor):

        def __call__(self, input_ids, scores):
            if self.rng is None:
                self.rng = torch.Generator()
            ids = [
                self._get_greenlist_ids(row.cpu()).to(scores.device)
                for row in input_ids
            ]
            mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=ids)
            return self._bias_greenlist_logits(
                scores=scores, greenlist_mask=mask, greenlist_bias=self.delta
            )

    return CPUSeededKGW(
        vocab=vocab, gamma=KGW_GAMMA, delta=KGW_DELTA, seeding_scheme="simple_1"
    )


class Scorer:

    def __init__(self, tokenizer):
        import torch
        from watermarking.kirchenbauer.watermark_processor import WatermarkDetector

        self.detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=KGW_GAMMA,
            seeding_scheme="simple_1",
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            z_threshold=1.5,
            normalizers=[],
            ignore_repeated_bigrams=False,
        )

    def __call__(self, tokens, seed, null=False, reference=False):
        return -float(self.detector._score_sequence(tokens)["z_score"])

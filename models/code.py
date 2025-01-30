from __future__ import annotations
import collections
from functools import lru_cache
import logging
import ipdb

import torch

from models.wllm import WatermarkDetector, WatermarkLogitsProcessor
from src.datahelper import ngrams

logger = logging.getLogger(__name__)

class CodeMixin:
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")
        
        self._seed_rng(input_ids, self.hash_key)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids[-self.context_width:])
            logits, switch = output['logits'], output['switch']
            switch = torch.sigmoid(switch)
            
            sorted_values, sorted_indices = torch.sort(logits, dim=1, descending=True)
            sorted_indices = sorted_indices.squeeze(0)
            seq_len = sorted_indices.shape[0]
            k = (seq_len + 1) // 2  # Ceiling division to handle odd length
            
            # Generate random choices for each pair and possibly the last element
            random_choices = torch.randint(0, 2, (k,), generator=self.rng, device=sorted_indices.device)
            
            # Create indices for pairs
            base_indices = torch.arange(0, seq_len-1, 2, device=sorted_indices.device)
            selected_indices = base_indices + random_choices[:len(base_indices)]
            
            # If sequence length is odd, potentially include the last element
            if seq_len % 2 == 1:
                if random_choices[-1].item():  # Use the last random choice for the last element
                    selected_indices = torch.cat([selected_indices, sorted_indices.new_tensor([seq_len-1])])
            
            greenlist_ids = sorted_indices[selected_indices]
            
            # Convert to list format for return
            greenlist_ids = greenlist_ids.cpu().tolist()
            switch = switch.squeeze(0).cpu().item()
        
        return greenlist_ids, switch
    

class CodeLogitsProcessor(CodeMixin, WatermarkLogitsProcessor):
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model", None)
        self.model.eval()
        self.context_width = kwargs.pop("context_width", 4)
        self.switch_threshold = kwargs.pop("switch_threshold", 0.0)
        logger.info(f"=== SWITCH_THRESHOLD: {self.switch_threshold} ===")
        super().__init__(*args, **kwargs)
        logger.info(f"=== hash key: {self.hash_key} ===")
    

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        switch_list = []
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids, switch = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids
            switch_list.append(switch)

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        switch_tensor = torch.tensor(switch_list, device=green_tokens_mask.device)
        switch_mask = (switch_tensor > self.switch_threshold).view(-1, 1)
        
        green_tokens_mask = green_tokens_mask * switch_mask
        
        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )
        return scores
        
class CodeDetector(CodeMixin, WatermarkDetector):
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model", None)
        self.model.eval()
        self.ignore_repeated_ngrams = kwargs.pop("ignore_repeated_ngrams", False)
        self.device = kwargs.pop("device", torch.device("cpu"))
        self.context_width = kwargs.pop("context_width", 4)
        self.switch_threshold = kwargs.pop("switch_threshold", 0.0)
        super().__init__(*args, **kwargs) 
        logger.info(f"=== hash key: {self.hash_key} ===")
        self.rng = torch.Generator(device=self.device)
    
    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        greenlist_ids, switch = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        green_bool = True if target in greenlist_ids else False
        return green_bool, switch
        
    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""

        # Compute scores for all ngrams contexts in the passage:
        token_ngram_generator = ngrams(input_ids.cpu().tolist(), self.context_width + 1)
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        switch_lookup = {}
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example[:-1]
            target = ngram_example[-1]
            output = self._get_ngram_score_cached(prefix, target)
            ngram_to_watermark_lookup[ngram_example] = output[0]
            switch_lookup[ngram_example] = output[1]

        return ngram_to_watermark_lookup, frequencies_table, switch_lookup

    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup, switch_lookup) -> tuple[torch.Tensor]:
        """Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of offsets to
        convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations where otherwise a repeat would be counted
        """
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        switch_mask, switch_mask_unique = [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            switch_mask.append(switch_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                    switch_mask_unique.append(switch_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                switch_mask_unique.append(switch_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
            torch.tensor(switch_mask),
            torch.tensor(switch_mask_unique),
        )
        
    def count_scores(self, input_ids, ngram_to_watermark_lookup, frequencies_table, green_unique):
        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width 
            green_token_count = sum(freq * outcome for freq, outcome in zip(frequencies_table.values(), ngram_to_watermark_lookup.values()))
        assert green_token_count == green_unique.sum()
        return num_tokens_scored, green_token_count

    def new_count_scores(self, input_ids, ngram_to_watermark_lookup, frequencies_table, green_unique, \
        switch_lookup, switch_mask, switch_mask_unique):
        filtered_ngrams = [ngram for ngram, switch in zip(ngram_to_watermark_lookup.keys(), switch_lookup.values()) if switch > self.switch_threshold]
        filtered_frequencies_table = {ngram: freq for ngram, freq in frequencies_table.items() if ngram in filtered_ngrams}
        filtered_ngrams_lookup = {ngram: ngram_to_watermark_lookup[ngram] for ngram in ngram_to_watermark_lookup if ngram in filtered_ngrams}
        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(filtered_frequencies_table.keys())
            green_token_count = sum(filtered_ngrams_lookup.values())
        else:
            num_tokens_scored = sum(filtered_frequencies_table.values())
            green_token_count = sum(freq * outcome for freq, outcome in zip(filtered_frequencies_table.values(), filtered_ngrams_lookup.values()))
        return num_tokens_scored, green_token_count


    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
        return_z_at_T: bool = True,
    ):
        
        ngram_to_watermark_lookup, frequencies_table, switch_lookup = self._score_ngrams_in_passage(input_ids)
        green_token_mask, green_unique, offsets, switch_mask, switch_mask_unique = self._get_green_at_T_booleans(input_ids, ngram_to_watermark_lookup, switch_lookup)

        # num_tokens_scored, green_token_count = self.count_scores(input_ids, ngram_to_watermark_lookup, frequencies_table, green_unique)
        num_tokens_scored, green_token_count = self.new_count_scores(input_ids, ngram_to_watermark_lookup, frequencies_table, green_unique, \
            switch_lookup, switch_mask, switch_mask_unique)
        
        score_dict = dict()
        if num_tokens_scored == 0:
            return {
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "green_fraction": 0,
                "z_score": 0.0,
                "p_value": 1,
            }
        
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(
                dict(green_fraction=(green_token_count / num_tokens_scored))
            )
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))
        return_z_at_T = False
        if return_z_at_T:
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

            score_dict.update(dict(z_score_at_T=z_score_at_T.cpu().tolist()))

        return score_dict

    def detect(
        self,
        tokenized_text: torch.Tensor = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        assert tokenized_text is not None, "Must pass either tokenized string"
    
        if return_prediction:
            kwargs["return_p_value"] = True

        if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
            tokenized_text = tokenized_text[1:]
            
        output_dict = {}
        
        if len(tokenized_text) < self.context_width <+1:
            output_dict["invalid"] = True
            return output_dict
        
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        
        
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]
                
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict


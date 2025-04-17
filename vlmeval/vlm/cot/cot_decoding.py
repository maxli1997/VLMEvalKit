import torch
from transformers import PreTrainedModel
from typing import List, Tuple, Dict, Optional
import numpy as np

def calculate_confidence(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.
    
    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer
    
    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            
            if top_2_probs.size(-1) > 1:
                #if (top_2_probs[-1][0] - top_2_probs[-1][1]).item() == 1:
                    #print(token_logits,torch.sum(token_logits))
                    #quit()
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0  # Max confidence if there's only one token
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1
    
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0

def aggregate_paths_based_on_scores(paths: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Aggregate multiple paths based on their confidence scores."""
    answer_scores = {}
    for answer, delta in paths:
        if answer == None:
            continue
        answer_scores[answer] = answer_scores.get(answer, 0) + delta
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_answer, answer_scores[best_answer]

def cot_decode_llama(
    model: PreTrainedModel,
    processor,
    inputs,
    image,
    generation_config: Dict,
    k_first: int = 10,
    n_depth: int = 2,
    k_other: int = 2,
    
) -> Tuple[str, float]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """

    '''# Use the chat template to format the input
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = tokenizer.from_list_format(messages)'''

    '''input_ids = tokenizer.encode(messages, return_tensors="pt").to(device)'''
    #attention_mask = torch.ones_like(input_ids).to(device)
    
    '''# Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id'''

    candidates = [inputs]
    with torch.no_grad():
        for i in range(n_depth):
            next_candidates = []
            for candidate in candidates:
                outputs = model.generate(
                    **candidate,
                    max_new_tokens=1,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    do_sample = generation_config.do_sample,
                    top_p = generation_config.top_p,
                    top_k = generation_config.top_k,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                first_token_logits = outputs.scores[0][-1]
                if i == 0:
                    _, top_k_indices = torch.topk(first_token_logits, k_first)
                else:
                    _, top_k_indices = torch.topk(first_token_logits, k_other)
                for idx in top_k_indices:
                    # Generate sequence starting with the selected token
                    start_ids = torch.cat([candidate.input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                    next_candidates.append(processor(image, processor.decode(start_ids[0],skip_special_tokens=False, clean_up_tokenization_spaces=False), return_tensors="pt").to('cuda'))
            candidates = next_candidates

    '''# Get the top-k tokens for the first decoding step
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)'''

    paths = []
    for candidate in candidates:
        output = model.generate(
            **candidate,
            max_new_tokens=generation_config.max_new_tokens,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            do_sample = generation_config.do_sample,
            top_p = generation_config.top_p,
            top_k = generation_config.top_k,
            output_scores=True,
            return_dict_in_generate=True,
        )

        
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(inputs.input_ids[0]):]
        answer_text = processor.decode(answer_ids, skip_special_tokens=True)
        
        # Calculate confidence score (Δ)
        confidence = calculate_confidence(output.scores, answer_ids)
        paths.append((answer_text, confidence, len(answer_ids) + n_depth))
    
    paths.sort(key=lambda x: x[1], reverse=True)
    return paths

def cot_decode_qwen_base(
    model: PreTrainedModel,
    processor,
    inputs,
    image,
    generation_config: Dict,
    k_first: int = 10,
    n_depth: int = 2,
    k_other: int = 2,
    
) -> Tuple[str, float]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """

    '''# Use the chat template to format the input
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = tokenizer.from_list_format(messages)'''

    '''input_ids = tokenizer.encode(messages, return_tensors="pt").to(device)'''
    #attention_mask = torch.ones_like(input_ids).to(device)
    
    '''# Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id'''

    candidates = [inputs]
    with torch.no_grad():
        for i in range(n_depth):
            next_candidates = []
            for candidate in candidates:
                outputs = model.generate(
                    **candidate,
                    max_new_tokens=1,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    do_sample = generation_config.do_sample,
                    top_p = generation_config.top_p,
                    top_k = generation_config.top_k,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                first_token_logits = outputs.scores[0][-1]
                if i == 0:
                    _, top_k_indices = torch.topk(first_token_logits, k_first)
                else:
                    _, top_k_indices = torch.topk(first_token_logits, k_other)
                for idx in top_k_indices:
                    # Generate sequence starting with the selected token
                    start_ids = torch.cat([candidate.input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                    out_text = processor.decode(start_ids[0],skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    out_text = out_text.replace('<|image_pad|>','').replace('<|vision_start|>','<|vision_start|><|image_pad|>')
                    next_candidates.append(processor(images=image, padding=True, text=[out_text], return_tensors="pt").to('cuda'))
            candidates = next_candidates

    '''# Get the top-k tokens for the first decoding step
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)'''

    paths = []
    for candidate in candidates:
        output = model.generate(
            **candidate,
            max_new_tokens=generation_config.max_new_tokens,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            do_sample = generation_config.do_sample,
            top_p = generation_config.top_p,
            top_k = generation_config.top_k,
            output_scores=True,
            return_dict_in_generate=True,
        )

        
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(inputs.input_ids[0]):]
        answer_text = processor.decode(answer_ids, skip_special_tokens=True)
        
        # Calculate confidence score (Δ)
        confidence = calculate_confidence(output.scores, answer_ids)
        paths.append((answer_text, confidence, len(answer_ids) + n_depth))
    
    paths.sort(key=lambda x: x[1], reverse=True)
    return paths

def cot_decode_qwen(
    model: PreTrainedModel,
    processor,
    inputs,
    image,
    video,
    generation_config: Dict,
    k_first: int = 4,
    n_depth: int = 1,
    k_other: int = 1,
    
) -> Tuple[str, float]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """

    '''# Use the chat template to format the input
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = tokenizer.from_list_format(messages)'''

    '''input_ids = tokenizer.encode(messages, return_tensors="pt").to(device)'''
    #attention_mask = torch.ones_like(input_ids).to(device)
    
    '''# Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id'''

    candidates = [inputs]
    with torch.no_grad():
        for i in range(n_depth):
            next_candidates = []
            for candidate in candidates:
                outputs = model.generate(
                    **candidate,
                    max_new_tokens=1,
                    do_sample = generation_config['do_sample'],
                    top_p = generation_config['top_p'],
                    top_k = generation_config['top_k'],
                    return_dict_in_generate=True,
                    output_scores=True
                )
                first_token_logits = outputs.scores[0][-1]
                if i == 0:
                    _, top_k_indices = torch.topk(first_token_logits, k_first)
                else:
                    _, top_k_indices = torch.topk(first_token_logits, k_other)
                for idx in top_k_indices:
                    # Generate sequence starting with the selected token
                    start_ids = torch.cat([candidate.input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                    out_text = processor.decode(start_ids[0],skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    out_text = out_text.replace('<|image_pad|>','').replace('<|vision_start|>','<|vision_start|><|image_pad|>')
                    next_candidates.append(processor(images=image, videos=video, padding=True, text=[out_text], return_tensors="pt").to('cuda'))
            candidates = next_candidates

    '''# Get the top-k tokens for the first decoding step
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)'''

    paths = []
    for candidate in candidates:
        output = model.generate(
            **candidate,
            max_new_tokens=generation_config['max_new_tokens'],
            do_sample = generation_config['do_sample'],
            top_p = generation_config['top_p'],
            top_k = generation_config['top_k'],
            output_scores=True,
            return_dict_in_generate=True,
        )

        
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(inputs.input_ids[0]):]
        answer_text = processor.decode(answer_ids, skip_special_tokens=True)
        
        # Calculate confidence score (Δ)
        confidence = calculate_confidence(output.scores, answer_ids)
        paths.append((answer_text, confidence, len(answer_ids) + n_depth))
    
    paths.sort(key=lambda x: x[1], reverse=True)
    return paths
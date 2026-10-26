from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from vllm.reasoning.muse_glimmer_reasoning_parser import (
    MuseGlimmerReasoningParser,
)
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.gemma4_utils import parse_thinking_output
from tqdm import tqdm
from functools import partial
import pandas as pd
from download_models import MODEL_HASHES
from pathlib import Path


# Model settings used vary a little to reflect their resource intensity and propensity for reasoning
MODEL_SETTINGS = {
    'qwen3.8': {
        'max_model_len': 3072,
        'tensor_parallel_size': 2, # Adapt depending on available resources
        'max_num_seqs': 64,
        'max_num_batched_tokens': 8192,
        'max_tokens': 3000,
        'reasoning_parser': 'qwen3',
        'chat_template_kwargs': {
            'preserve_thinking': False,
            'enable_thinking': True,
            'reasoning_effort': 'low',
        },
    },
    'gemma4': {
        'max_model_len': 16384,
        'tensor_parallel_size': 4,
        'max_num_seqs': 96,
        'max_num_batched_tokens': 16384,
        'max_tokens': 15_000,
        'reasoning_parser': 'gemma4',
        'chat_template_kwargs': {
            'preserve_thinking': False,
            'enable_thinking': True,
            'reasoning_effort': 'low',
        },
    },
    'muse': {
        'max_model_len': 16384,
        'tensor_parallel_size': 2,
        'max_num_seqs': 96,
        'max_num_batched_tokens': 16384,
        'max_tokens': 15_000,
        'reasoning_parser': 'muse_glimmer',
        'chat_template_kwargs': {'reasoning_strength': 'low'},
    },
}



def run_games(
            prompts, 
            temperature,
            model,
            model_root,
            random_seed_start=4321,
            *,
            condition,
            benefit,
            cost,
            public_belief_proportion,
):
    
    settings = MODEL_SETTINGS[model]
    version_hash = MODEL_HASHES[model]
    model_dir = Path(model_root) / model / f"{model}-{version_hash}"

    llm_kwargs = {}
    if model != 'qwen3.8':
        llm_kwargs['trust_remote_code'] = True
        llm_kwargs['limit_mm_per_prompt'] = {'image': 0, 'audio': 0}

    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        max_model_len=settings['max_model_len'],
        tensor_parallel_size=settings['tensor_parallel_size'],
        enable_prefix_caching=True,
        async_scheduling=model != 'qwen3.8',
        max_num_seqs=settings['max_num_seqs'],
        max_num_batched_tokens=settings['max_num_batched_tokens'],
        gpu_memory_utilization=0.95,
        disable_log_stats=False,
        enforce_eager=True,
        reasoning_config=ReasoningConfig(
            reasoning_parser=settings['reasoning_parser']
        ),
        **llm_kwargs,
    )

    conversations = [
        [{'role': 'user', 'content': prompt}]
        for prompt in prompts
    ]

    sampling_params = [
        SamplingParams(
            max_tokens=settings['max_tokens'],
            temperature=temperature,
            seed=random_seed_start + i,
            skip_special_tokens=model == 'gemma4',
        )
        for i in range(len(conversations))
    ]

    results = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs=settings['chat_template_kwargs'],
        use_tqdm=partial(tqdm, mininterval=60),
    )

    raw_output = [result.outputs[0].text for result in results]

    if model == 'muse':
        parser = MuseGlimmerReasoningParser(llm.get_tokenizer())
        parsed = [
            parser.extract_reasoning(text, request=None)
            for text in raw_output
        ]
        reasoning = [item[0] for item in parsed]
        answer = [item[1] for item in parsed]

    elif model == 'gemma4':
        parsed = [parse_thinking_output(text) for text in raw_output]
        reasoning = [item['thinking'] for item in parsed]
        answer = [item['answer'] for item in parsed]

    elif model == 'qwen3.8':
        parser_class = ReasoningParserManager.get_reasoning_parser('qwen3')
        parser = parser_class(
            llm.get_tokenizer(),
            chat_template_kwargs=settings['chat_template_kwargs'],
        )
        parsed = [
            parser.extract_reasoning(text, request=None)
            for text in raw_output
        ]
        reasoning = [item[0] for item in parsed]
        answer = [item[1] for item in parsed]

    return pd.DataFrame(
        {
            'condition': condition,
            'benefit': benefit,
            'cost': cost,
            'public_belief_proportion': public_belief_proportion,
            'sample_temperature': temperature,
            'prompt': prompts,
            'raw_output': raw_output,
            'reasoning': reasoning,
            'answer': answer,
            'model': model,
            'version_hash': version_hash
        }
    )
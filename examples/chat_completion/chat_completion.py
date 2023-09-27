# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import json

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import fire
import torch
import os
import sys
import warnings
from typing import List

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker


def main(
        model_name,
        force_cn: bool = False,
        ret_file: str = None,
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=256,  # The maximum numbers of tokens to generate
        min_new_tokens: int = 0,  # The minimum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        safety_score_threshold: float = 0.5,
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
        enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
        enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
        enable_saleforce_content_safety: bool = True,  # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs = read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print(f"User dialogs:\n{dialogs}")
    print("\n==================================\n")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)
        print(f"====model {model}")
        model.print_trainable_parameters()
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    chats, chats_raw = format_tokens(dialogs, tokenizer)
    logits_processor=None
    if force_cn:
        import re
        def is_chinese(word):
            """
            判断一个字符串是否为汉字
            """
            if re.match('[\u4e00-\u9fff]', word):
                return True
            else:
                return False

        CHINESE_TOKEN_IDS = [token_id for token, token_id in tokenizer.get_vocab().items() if is_chinese(token)]
        from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
        class ChineseLogitsProcessor(LogitsProcessor):
            """
            生成中文本符
            ---------------
            ver: 2023-08-02
            by: changhongyu
            """

            def __init__(self,
                         chinese_token_id_list: List[int] = None,
                         alpha: float = 5):
                """
                :param chinese_token_id_list: 中文token的token的id列表
                :param alpha: 放大倍数
                """
                self.chinese_token_id_list = chinese_token_id_list
                self.alpha = alpha

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                for id_ in self.chinese_token_id_list:
                    scores[:, id_] *= self.alpha
                return scores

        CHINESE_TOKEN_IDS.append(tokenizer.eos_token_id)
        puncs = ['，', '。', '？', '！', '“', '”', '：', ',', '.', '?', '!', '"', "'", ':']
        CHINESE_TOKEN_IDS.extend(tokenizer.convert_tokens_to_ids(puncs))
        logits_processor = LogitsProcessorList()
        logits_processor.append(ChineseLogitsProcessor(CHINESE_TOKEN_IDS))
        print("====force cn====")

    with torch.no_grad():
        with open(ret_file, 'w') as fout:
            for idx, chat in enumerate(chats):
                print(f"==========chat {idx}=========")
                # safety_checker = get_safety_checker(enable_azure_content_safety,
                #                                     enable_sensitive_topics,
                #                                     enable_saleforce_content_safety,
                #                                     )
                # # Safety check of the user prompt
                # safety_results = [check(dialogs[idx][0]["content"]) for check in safety_checker]
                # are_safe = all([r[1] for r in safety_results])
                # if are_safe:
                #     print(f"User prompt deemed safe.")
                #     print("User prompt:\n", dialogs[idx][0]["content"])
                #     print("\n==================================\n")
                # else:
                #     print("User prompt deemed unsafe.")
                #     for method, is_safe, report in safety_results:
                #         if not is_safe:
                #             print(method)
                #             print(report)
                #     print("Skipping the inferece as the prompt is not safe.")
                #     sys.exit(1)  # Exit the program with an error status

                tokens = torch.tensor(chat).long()
                tokens = tokens.unsqueeze(0)
                tokens = tokens.to("cuda:0")
                if logits_processor:
                    outputs = model.generate(
                        tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        use_cache=use_cache,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        logits_processor=logits_processor,
                        **kwargs
                    )
                else:
                    outputs = model.generate(
                        input_ids=tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        use_cache=use_cache,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        **kwargs
                    )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Model output:\n{output_text}")
                req_len = sum([len(x) for x in chats_raw[idx]]) + 1
                item = {
                    'prompt': dialogs[idx],
                    'ans': output_text[req_len:].strip(),
                }
                fout.write(f"{json.dumps(item, ensure_ascii=False)}\n")
                # Safety check of the model output
                # safety_results = [check(output_text) for check in safety_checker]
                # are_safe = all([r[1] for r in safety_results])
                # if are_safe:
                #     print("User input and model output deemed safe.")
                #     print(f"Model output:\n{output_text}")
                #     print("\n==================================\n")
                #
                # else:
                #     print("Model output deemed unsafe.")
                #     for method, is_safe, report in safety_results:
                #         if not is_safe:
                #             print(method)
                #             print(report)


if __name__ == "__main__":
    fire.Fire(main)

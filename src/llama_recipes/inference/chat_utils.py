# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Literal, Optional, Tuple, TypedDict, Union
import json



Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    prompt_txt = []
    for i, dialog in enumerate(dialogs):
        if dialog[0]["role"] == "system":
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS
                                        + dialog[0]["content"]
                                        + E_SYS
                                        + dialog[1]["content"],
                         }
                     ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )

        assert (
                dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        dialog_txt = [f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} " for prompt, answer in zip(dialog[::2], dialog[1::2])]
        dialog_txt.append(f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}")
        prompt_txt.append(dialog_txt)
        prompt_tokens.append(dialog_tokens)
        # print(f"==========dialog_txt {i}=======")
        # print(json.dumps(dialog_txt, ensure_ascii=False, indent=2))
    return prompt_tokens, prompt_txt


def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs

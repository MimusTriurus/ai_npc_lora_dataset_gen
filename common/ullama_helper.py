import ctypes
import json
import time
from typing import Optional, Tuple, List, Dict

from ullama_python.ullama_python.ullama import ULlamaWrapper, split_think_and_json

from common.helpers import replace_unicode


class ULlamaHelper:
    """
    Wrapper over ULlamaWrapper that mirrors the OllamaHelper.chat() interface.

    Differences from OllamaHelper:
    - __init__ accepts a base config dict instead of a host string.
    - The model/lora paths live in the config; the `model` arg in chat()
      overrides config["model"] and triggers a reload when it changes.
    - system_prompt in chat() overrides config["system_prompt"] and
      triggers a worker re-initialisation when it changes.
    - history is serialised as a plain-text prefix appended to user_prompt
      (the underlying engine has no native multi-turn API).
    - Returns (answer_text, think_block) — same shape as OllamaHelper.
    """

    TOKEN_BUF_SIZE = 16 * 1024

    def __init__(self, base_config: dict, init_chat: bool = True) -> None:
        """
        Parameters
        ----------
        base_config : dict
            Chat config. Must contain at least:
              "model"        – path to the .gguf base model
              "lora_adapter" – path to the LoRA adapter .gguf
            Optional keys forwarded to the engine verbatim:
              "grammar", "temperature", etc.
        init_chat : bool
            When False, no chat model/worker is loaded. Use this for
            embedding-only flows that only need the knowledge base.
        """
        self._base_config: dict = dict(base_config)

        self._api: Optional[ULlamaWrapper] = None

        # Chat resources (own model + worker, driven by base_config)
        self._model = None
        self._worker = None

        # Knowledge-base resources (own config + own embedding model + worker)
        self._kb_config: Optional[dict] = None
        self._kb_model = None
        self._kb_worker = None

        self._token_buf = ctypes.create_string_buffer(self.TOKEN_BUF_SIZE)

        # Track what is currently loaded so we know when to reload
        self._loaded_model_path: Optional[str] = None
        self._loaded_system_prompt: Optional[str] = None

        if init_chat:
            self._load_model(
                model_path=self._base_config.get("model", ""),
                system_prompt=self._base_config.get("system_prompt", ""),
            )
        else:
            # Embedding-only flow: we still need an API handle so the KB
            # can be initialised later via kb_init(...).
            self._api = ULlamaWrapper()

    # ------------------------------------------------------------------
    # Public API (mirrors OllamaHelper)
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[dict]] = None,
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Parameters
        ----------
        model         : model file path (overrides base_config["model"]).
        system_prompt : overrides base_config["system_prompt"]; triggers
                        worker re-init when it differs from the loaded one.
        user_prompt   : the user's current message (raw string or JSON str).
        history       : optional list of {"role": ..., "content": ...} dicts
                        — serialised as a text prefix before user_prompt.

        Returns
        -------
        (answer, thinking) – both Optional[str], same as OllamaHelper.
        """
        try:
            # Reload model / reinit worker only when something changed
            if model and model != self._loaded_model_path:
                self._load_model(model, system_prompt)
                print(f'[WARNING] Reloading model "{model}"')
            elif system_prompt != self._loaded_system_prompt:
                self._reinit_worker(system_prompt)
                print(f'[WARNING] Reloading worker "{model}"')

            full_prompt = self._build_prompt(user_prompt, history)
            raw_response = self._ask(full_prompt)

            raw_response = replace_unicode(raw_response)
            think_block, response_dict = split_think_and_json(raw_response)

            if response_dict is None:
                print(f"[ERR] ullama chat: can't parse response: {raw_response}")
                return None, think_block

            return response_dict, think_block

        except Exception as e:
            print(f"[ERR] unexpected: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Knowledge base API (mirrors step_2_validate_knowledge_base/main.py)
    # ------------------------------------------------------------------

    def kb_init(self, kb_config: dict, chunks: List[str]) -> None:
        """
        Build a knowledge base. The KB uses its OWN config and loads its own
        embedding model — independent of the chat model/config.

        Parameters
        ----------
        kb_config : embedding-model config (own "model"/"lora_adapter"/etc.),
                    same shape as the inference_cfg used in
                    step_2_validate_knowledge_base/main.py.
        chunks    : plain-text passages (e.g. action signatures) to index.
        """
        if self._api is None:
            self._api = ULlamaWrapper()

        # Drop any previous KB (worker + embedding model)
        self._dispose_kb()

        self._kb_config = dict(kb_config)
        cfg_bytes = json.dumps(self._kb_config).encode("utf-8")

        self._kb_model = self._api.lib.ullama_load_model(cfg_bytes)
        if not self._kb_model:
            self._dispose_kb()
            raise RuntimeError("kb_init: ullama_load_model failed — check KB config")

        self._kb_worker = self._api.lib.ullama_kb_make()
        if not self._api.lib.ullama_kb_init(self._kb_worker, cfg_bytes, self._kb_model):
            self._dispose_kb()
            raise RuntimeError("ullama_kb_init failed — check KB model / config paths")

        for chunk_text in chunks:
            self._api.lib.ullama_kb_add_chunk(
                self._kb_worker, chunk_text.encode("utf-8")
            )
        self._api.lib.ullama_kb_update(self._kb_worker)

    def kb_add_chunk(self, chunk_text: str) -> None:
        if self._kb_worker is None:
            raise RuntimeError("kb_add_chunk: knowledge base is not initialised")
        self._api.lib.ullama_kb_add_chunk(
            self._kb_worker, chunk_text.encode("utf-8")
        )

    def kb_update(self) -> None:
        if self._kb_worker is None:
            raise RuntimeError("kb_update: knowledge base is not initialised")
        self._api.lib.ullama_kb_update(self._kb_worker)

    def kb_search(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """Return up to top_k (chunk_index, score) pairs for the query."""
        if self._kb_worker is None:
            raise RuntimeError("kb_search: knowledge base is not initialised")
        return self._api.search_top_n(
            kb_handle=self._kb_worker,
            query=query,
            top_k=top_k,
        )

    def kb_dispose(self) -> None:
        self._dispose_kb()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dispose_kb(self) -> None:
        """Release the KB worker and its embedding model (chat is untouched)."""
        if self._api is None:
            self._kb_worker = None
            self._kb_model = None
            return
        if self._kb_worker is not None:
            self._api.lib.ullama_kb_dispose(self._kb_worker)
            self._kb_worker = None
        if self._kb_model is not None:
            self._api.lib.ullama_free_model(self._kb_model)
            self._kb_model = None


    def _build_config(self, model_path: str, system_prompt: str) -> dict:
        cfg = dict(self._base_config)
        if model_path:
            cfg["model"] = model_path
        if system_prompt is not None:
            cfg["system_prompt"] = system_prompt
        return cfg

    def _load_model(self, model_path: str, system_prompt: str) -> None:
        self._teardown()

        cfg = self._base_config #self._build_config(model_path, system_prompt)
        cfg_bytes = json.dumps(cfg).encode("utf-8")
        self._api = ULlamaWrapper()
        self._model = self._api.lib.ullama_load_model(cfg_bytes)
        self._worker = self._api.lib.ullama_make()
        if not self._api.lib.ullama_init(self._worker, cfg_bytes, self._model):
            self._teardown()
            raise RuntimeError("ullama_worker_init failed — check model / config paths")

        self._api.lib.ullama_run(self._worker)
        self._loaded_model_path = cfg["model"]
        self._loaded_system_prompt = system_prompt

    def _reinit_worker(self, system_prompt: str) -> None:
        """Re-initialise only the worker (cheaper than reloading the model)."""
        if self._worker is not None:
            self._api.lib.ullama_dispose(self._worker)

        cfg = self._build_config(self._loaded_model_path, system_prompt)
        cfg_bytes = json.dumps(cfg).encode("utf-8")

        self._worker = self._api.lib.ullama_make()
        if not self._api.lib.ullama_init(self._worker, cfg_bytes, self._model):
            self._worker = None
            raise RuntimeError("ullama_worker_init failed during reinit")

        self._api.lib.ullama_run(self._worker)
        self._loaded_system_prompt = system_prompt

    def _ask(self, prompt: str) -> str:
        """Send a single prompt and block until the worker delivers a complete response.

        Mirrors the C++ ask_and_wait: while the worker is speaking we wait;
        once it stops we pull the full response in one call. The trailing
        sleep handles the window between ullama_ask and the start of
        generation, where is_speaking is still false but no response exists
        yet — without it we would return "" and leave the real response
        buffered, which then gets concatenated onto the next request.
        """
        self._api.lib.ullama_ask(self._worker, prompt.encode("utf-8"))
        while True:
            if self._api.lib.ullama_is_speaking(self._worker):
                time.sleep(0.001)
                continue
            if self._api.lib.ullama_get_response(
                self._worker, self._token_buf, self.TOKEN_BUF_SIZE
            ):
                return self._token_buf.value.decode("utf-8")
            time.sleep(0.001)

    @staticmethod
    def _build_prompt(user_prompt: str, history: Optional[List[dict]]) -> str:
        """
        Prepend history to the user prompt as plain text.

        History format (same as OllamaHelper):
            [{"role": "user"|"assistant"|"system", "content": "..."}]
        """
        if not history:
            return user_prompt

        lines: List[str] = []
        for msg in history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        lines.append(f"User: {user_prompt}")
        return "\n".join(lines)

    def _teardown(self) -> None:
        """Release native resources if they exist."""
        if self._api is None:
            return
        self._dispose_kb()
        if self._worker is not None:
            self._api.lib.ullama_dispose(self._worker)
            self._worker = None
        if self._model is not None:
            self._api.lib.ullama_free_model(self._model)
            self._model = None
        self._api = None

    def __del__(self) -> None:
        self._teardown()
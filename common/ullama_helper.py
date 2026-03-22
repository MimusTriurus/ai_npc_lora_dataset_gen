import ctypes
import json
from typing import Optional, Tuple, List, Dict

from ullama_python.ullama import ULlamaWrapper, split_think_and_json

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

    TOKEN_BUF_SIZE = 512

    def __init__(self, base_config: dict) -> None:
        """
        Parameters
        ----------
        base_config : dict
            Must contain at least:
              "model"        – path to the .gguf base model
              "lora_adapter" – path to the LoRA adapter .gguf
            Optional keys forwarded to the engine verbatim:
              "grammar", "temperature", etc.
        """
        self._base_config: dict = dict(base_config)

        self._api: Optional[ULlamaWrapper] = None
        self._model = None
        self._worker = None
        self._token_buf = ctypes.create_string_buffer(self.TOKEN_BUF_SIZE)

        # Track what is currently loaded so we know when to reload
        self._loaded_model_path: Optional[str] = None
        self._loaded_system_prompt: Optional[str] = None

        self._load_model(
            model_path=self._base_config.get("model", ""),
            system_prompt=self._base_config.get("system_prompt", ""),
        )

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
            elif system_prompt != self._loaded_system_prompt:
                self._reinit_worker(system_prompt)

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
    # Private helpers
    # ------------------------------------------------------------------

    def _build_config(self, model_path: str, system_prompt: str) -> dict:
        cfg = dict(self._base_config)
        if model_path:
            cfg["model"] = model_path
        if system_prompt is not None:
            cfg["system_prompt"] = system_prompt
        return cfg

    def _load_model(self, model_path: str, system_prompt: str) -> None:
        """Tear down any existing model+worker and load a new one."""
        self._teardown()

        cfg = self._build_config(model_path, system_prompt)
        cfg_bytes = json.dumps(cfg).encode("utf-8")

        self._api = ULlamaWrapper()
        self._model = self._api.lib.ullama_loadModel(cfg_bytes)
        self._worker = self._api.lib.ullama_worker_make()

        if not self._api.lib.ullama_worker_init(self._worker, cfg_bytes, self._model):
            self._teardown()
            raise RuntimeError("ullama_worker_init failed — check model / config paths")

        self._api.lib.ullama_worker_run(self._worker)
        self._loaded_model_path = cfg["model"]
        self._loaded_system_prompt = system_prompt

    def _reinit_worker(self, system_prompt: str) -> None:
        """Re-initialise only the worker (cheaper than reloading the model)."""
        if self._worker is not None:
            self._api.lib.ullama_worker_dispose(self._worker)

        cfg = self._build_config(self._loaded_model_path, system_prompt)
        cfg_bytes = json.dumps(cfg).encode("utf-8")

        self._worker = self._api.lib.ullama_worker_make()
        if not self._api.lib.ullama_worker_init(self._worker, cfg_bytes, self._model):
            self._worker = None
            raise RuntimeError("ullama_worker_init failed during reinit")

        self._api.lib.ullama_worker_run(self._worker)
        self._loaded_system_prompt = system_prompt

    def _ask(self, prompt: str) -> str:
        """Send a single prompt and collect tokens until the worker is done."""
        self._api.lib.ullama_worker_ask(self._worker, prompt.encode("utf-8"))
        response = ""
        while self._api.lib.ullama_worker_isSpeaking(self._worker):
            if self._api.lib.ullama_worker_getToken(
                self._worker, self._token_buf, self.TOKEN_BUF_SIZE
            ):
                response += self._token_buf.value.decode("utf-8")
        return response

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
        if self._worker is not None:
            self._api.lib.ullama_worker_dispose(self._worker)
            self._worker = None
        if self._model is not None:
            self._api.lib.ullama_freeModel(self._model)
            self._model = None
        self._api = None

    def __del__(self) -> None:
        self._teardown()

# from pathlib import Path
# MODEL_PATH = Path("../model/llama-2-7b-chat.Q4_K_M.gguf")

# class LocalLLM:
#     def __init__(self, model_path=str(MODEL_PATH)):
#         # tentative d'import : choisir l'API installée
#         try:
#             from llama_cpp import Llama
#             # n_ctx = taille du contexte, ajuster si besoin
#             self.model = Llama(model_path=model_path, n_ctx=512)
#             self.backend = "llama_cpp"
#         except Exception:
#             # fallback : GPT4All
#             try:
#                 from gpt4all import GPT4All
#                 self.model = GPT4All(model="gpt4all-lora-quantized")  # exemple
#                 self.backend = "gpt4all"
#             except Exception:
#                 raise RuntimeError("Aucun backend LLM disponible. Installer 'llama-cpp-python' ou 'gpt4all'.")

#     def generate(self, prompt, max_tokens=256, temperature=0.1):
#         if self.backend == "llama_cpp":
#             # Nouvelle API llama-cpp : on appelle l'objet directement
#             resp = self.model(
#                 prompt=prompt,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 stop=["</s>"]
#             )
#             return resp["choices"][0]["text"]
#         elif self.backend == "gpt4all":
#             out = self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
#             return out
from pathlib import Path
MODEL_PATH = Path("../model/llama-2-7b-chat.Q4_K_M.gguf")

class LocalLLM:
    def __init__(self, model_path=str(MODEL_PATH)):
        try:
            from llama_cpp import Llama
            # on met une fenêtre plus grande (2048 ou 4096 si votre PC tient)
            self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
            self.backend = "llama_cpp"
        except Exception:
            try:
                from gpt4all import GPT4All
                self.model = GPT4All(model="gpt4all-lora-quantized")
                self.backend = "gpt4all"
            except Exception:
                raise RuntimeError("Aucun backend LLM disponible. Installer 'llama-cpp-python' ou 'gpt4all'.")

    def generate(self, prompt, max_tokens=256, temperature=0.1):
        if self.backend == "llama_cpp":
            resp = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>"]
            )
            return resp["choices"][0]["text"]
        elif self.backend == "gpt4all":
            return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)

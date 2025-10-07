# # app.py
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from retriever import Retriever
# from llm import LocalLLM
# from pathlib import Path
# import textwrap

# app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")
# CORS(app)

# retriever = Retriever()
# llm = LocalLLM()

# # Template prompt builder for RAG
# def build_prompt(question, contexts):
#     header = "Tu es un assistant pédagogique spécialisé en aviation. Réponds de manière claire et concise en français. Utilise uniquement les informations présentes dans les passages fournis. Si tu dois deviner, signale clairement l'incertitude.\n\n"
#     ctx_texts = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in contexts])
#     prompt = f"{header}Contextes pertinents :\n{ctx_texts}\n\nQuestion: {question}\n\nRéponse:"
#     # limiter taille si nécessaire
#     return prompt

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     question = data.get("question", "")
#     top_k = int(data.get("top_k", 5))
#     if not question:
#         return jsonify({"error":"missing question"}), 400
#     contexts = retriever.retrieve(question, top_k=top_k)
#     prompt = build_prompt(question, contexts)
#     answer = llm.generate(prompt, max_tokens=400)
#     # return contexts (sources) and answer
#     return jsonify({"answer": answer, "sources": contexts})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
# app.py
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from retriever import Retriever
from llm import LocalLLM

# configure logger simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# compute absolute template/static folders relative to this file
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BASE_DIR.parent / "frontend")
TEMPLATE_FOLDER = str(FRONTEND_DIR / "templates")
STATIC_FOLDER = str(FRONTEND_DIR / "static")

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
CORS(app)

# instantiate retriever and llm (peuvent lever si mal configurés)
retriever = Retriever()
llm = LocalLLM()


def build_prompt(question, contexts, max_chars=3000):
    """
    Construit le prompt RAG en limitant la longueur texte concaténé.
    """
    header = (
        "Tu es un assistant pédagogique spécialisé en aviation. "
        "Réponds de manière claire et concise en français. "
        "Utilise uniquement les informations présentes dans les passages fournis. "
        "Si tu dois deviner, signale clairement l'incertitude.\n\n"
    )
    ctx_texts = "\n\n---\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in contexts])
    # truncation simple pour éviter de dépasser la fenêtre de contexte
    if len(ctx_texts) > max_chars:
        ctx_texts = ctx_texts[:max_chars] + "\n\n[...contexte tronqué...]"
    prompt = f"{header}Contextes pertinents :\n{ctx_texts}\n\nQuestion: {question}\n\nRéponse:"
    return prompt


@app.route("/")
def index():
    """
    Renvoie le fichier index.html. Si le fichier n'existe pas, renvoie un message clair.
    """
    index_path = Path(TEMPLATE_FOLDER) / "index.html"
    if not index_path.exists():
        logger.error("index.html introuvable dans %s", TEMPLATE_FOLDER)
        return (
            "<h1>404 - Frontend introuvable</h1>"
            "<p>Placez <code>index.html</code> dans frontend/templates/ puis relancez.</p>",
            404,
        )
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    top_k = int(data.get("top_k", 3))

    if not question:
        return jsonify({"error": "missing question"}), 400

    # sécurité: limiter top_k à une petite valeur
    top_k = max(1, min(top_k, 5))

    try:
        contexts = retriever.retrieve(question, top_k=top_k)
    except Exception as e:
        logger.exception("Erreur lors de la récupération des contextes")
        return jsonify({"error": "retriever error", "details": str(e)}), 500

    prompt = build_prompt(question, contexts, max_chars=3000)

    try:
        answer = llm.generate(prompt, max_tokens=400)
    except ValueError as e:
        # erreurs du type "Requested tokens exceed context window"
        logger.exception("LLM ValueError")
        return (
            jsonify(
                {
                    "error": "LLM error - fenêtre de contexte",
                    "details": str(e),
                    "hint": "Réduis le nombre de sources (top_k) ou tronque le contexte.",
                }
            ),
            400,
        )
    except Exception as e:
        logger.exception("Erreur durant la génération LLM")
        return jsonify({"error": "llm error", "details": str(e)}), 500

    return jsonify({"answer": answer, "sources": contexts})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # debug activé pour dev local; désactiver en production
    app.run(host="0.0.0.0", port=5000, debug=True)

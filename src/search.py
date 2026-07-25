import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
GOOGLE_LLM_MODEL = "gemini-3.1-flash-lite"

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    # Monta os componentes reutilizáveis uma única vez.
    # Se algo falhar aqui (banco fora do ar, chave inválida, etc.),
    # devolvemos None em vez de deixar a exceção propagar, pra quem
    # chamou (o chat.py) poder tratar com uma mensagem amigável.
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

        store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )

        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,  # resposta determinística, sem "criatividade"
        )
    except Exception as e:
        print(f"Erro ao inicializar componentes de busca: {e}")
        return None

    def ask(pergunta: str) -> str:
        # 1. Vetoriza a pergunta e busca os k=10 chunks mais relevantes
        resultados = store.similarity_search_with_score(pergunta, k=10)

        # 2. Concatena o texto dos chunks recuperados como CONTEXTO
        contexto = "\n\n".join(doc.page_content for doc, score in resultados)

        # 3. Monta o prompt final e chama a LLM
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
        resposta = llm.invoke(prompt)

        # 4. Retorna só o texto da resposta
        return resposta.content

    # Se já veio uma pergunta, responde direto (útil pra testar o search.py isolado)
    if question:
        return ask(question)

    # Sem pergunta: devolve a função pronta pra ser reusada pelo chat.py
    return ask


if __name__ == "__main__":
    print(search_prompt("Isso é um teste, o que você consegue ver no contexto?"))
# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e busca semântica sobre um PDF usando LangChain, PostgreSQL + pgVector e a API do Gemini (Google).

## Arquitetura

- `src/ingest.py`: lê o PDF, quebra em chunks (1000 caracteres, overlap de 150), gera embeddings e grava no Postgres via pgVector.
- `src/search.py`: monta a busca vetorial (`similarity_search_with_score`, k=10) + chamada à LLM com o prompt que restringe a resposta ao contexto recuperado.
- `src/chat.py`: CLI que faz o loop de perguntas e respostas usando `search.py`.

## Pré-requisitos

- Python 3.13+
- Docker e Docker Compose
- Uma API key do Google AI Studio (https://aistudio.google.com/app/apikey)

## Configuração

1. Clone o repositório e entre na pasta do projeto.

2. Crie e ative o ambiente virtual:

   ```
   python -m venv venv
   # Windows
   venv\Scripts\Activate.ps1
   # Linux/Mac
   source venv/bin/activate
   ```

3. Instale as dependências:

   ```
   pip install -r requirements.txt
   ```

4. Copie o `.env.example` para `.env` e preencha:

   ```
   GOOGLE_API_KEY=sua_chave_aqui
   GOOGLE_EMBEDDING_MODEL='models/gemini-embedding-001'
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
   PG_VECTOR_COLLECTION_NAME=document_chunks
   PDF_PATH=document.pdf
   ```

   > Nota: o enunciado original sugere `models/embedding-001`, mas esse modelo foi descontinuado pelo Google. Use `gemini-embedding-001` no lugar. O mesmo vale para o modelo de LLM em `src/search.py` (`gemini-3.1-flash-lite` no lugar de `gemini-3.1-flash-lite-preview`, que foi desativado).

## Execução

1. Suba o banco de dados:

   ```
   docker compose up -d
   ```

2. Rode a ingestão do PDF (só precisa rodar uma vez, ou de novo se trocar o PDF — o script limpa a coleção antes de reingerir):

   ```
   python src/ingest.py
   ```

3. Rode o chat:

   ```
   python src/chat.py
   ```

4. Digite suas perguntas no prompt `PERGUNTA:`. Uma linha vazia encerra o chat.

## Exemplo

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

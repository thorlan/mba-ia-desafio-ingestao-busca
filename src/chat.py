from search import search_prompt

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta:\n")

    while True:
        pergunta = input("PERGUNTA: ").strip()

        if not pergunta:
            print("Encerrando o chat.")
            break

        resposta = chain(pergunta)
        print(f"RESPOSTA: {resposta}\n")

if __name__ == "__main__":
    main()
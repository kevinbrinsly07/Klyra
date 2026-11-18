
from app.services.llm_service import llm_service


def main():

    print("LankaMind AI - OpenAI Model Test")
    print("--------------------------------")

    while True:
        prompt = input("\nYou: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        try:
            answer = llm_service.generate_response(prompt)
            print(f"\nLankaMind AI: {answer}")

        except Exception as error:
            print(f"\nModel error: {error}")


if __name__ == "__main__":
    main()

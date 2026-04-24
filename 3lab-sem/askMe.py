import requests

API_URL = "http://127.0.0.1:8000/generate"

def ask(question: str):

    response = requests.post(
        API_URL,
        json={"message": question}
    )

    data = response.json()
    print(f"\nВопрос: {question}")
    print(f"Ответ: {data.get('answer')}")


if __name__ == "__main__":
    questions = [
        "Сделай контент-план на 7 дней для фитнес тренера в CSV",
        "Что ты можешь посоветовать, чтобы раскрутить свой телеграм канал?",
        "Какие тренды в маркетинге сейчас?",
        "Какая сегодня погода?",
        "Привет, как дела?"
    ]

    for q in questions:
        ask(q)
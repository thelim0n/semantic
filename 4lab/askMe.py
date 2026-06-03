import requests

API_URL = "http://127.0.0.1:8000/generate"

def ask(question: str):

    response = requests.post(
        API_URL,
        json={"message": question}
    )

    data = response.json()
    print(f"\nВопрос: {question}")
    print(f"Ответ: {data}")


if __name__ == "__main__":
    questions = [
        "Какие тренды сейчас популярны в TikTok маркетинге?",
        "Сделай контент-план на 7 дней для фитнес-тренера в CSV"
    ]

    for q in questions:
        ask(q)
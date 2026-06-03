import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from ddgs import DDGS

load_dotenv()


# =========================
# TOOLS
# =========================

@tool
def search(query: str) -> str:
    """Поиск в интернете. Использовать только для трендов."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "Ничего не найдено"

    return "\n".join(
        f"{r['title']}: {r['body']}"
        for r in results
    )


@tool
def create_csv(content: str, filename: str) -> str:
    """
    Создает CSV файл.

    Требования:
    - разделитель ;
    - строки через \n
    """

    os.makedirs("data", exist_ok=True)

    path = os.path.join("data", filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())

    return f"CSV создан: {path}"


# =========================
# MODEL
# =========================

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

tools = [
    search,
    create_csv,
]

# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
Ты профессиональный маркетолог и AI-агент.

Твои задачи:
- создавать контент-планы
- генерировать идеи постов
- анализировать ЦА
- находить тренды

Инструменты:

1. search(query)
Используй только для трендов и актуальной информации.

2. create_csv(content, filename)
Используй только если пользователь просит:
- csv
- таблицу
- контент-план
- файл

Правила:

Если нужен CSV:
- создай CSV
- вызови create_csv
- не используй другие инструменты

Если нужны тренды:
- вызови search один раз
- кратко ответь результатами

Если обычный маркетинговый вопрос:
- отвечай текстом
- инструменты не используй

Если запрос не связан с маркетингом:
отвечай:
"Я работаю только с маркетингом"
"""

# =========================
# AGENT
# =========================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

# =========================
# PUBLIC API
# =========================

def generate_answer(user_input: str) -> dict:
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        }
    )

    messages = result["messages"]

    answer = messages[-1].content

    return {
        "answer": answer
    }
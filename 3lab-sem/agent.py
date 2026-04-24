import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from ddgs import DDGS

load_dotenv()

# tools
@tool
def search(query: str) -> str:
    """Поиск в интернете. Использовать только для трендов."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])

@tool
def create_csv(content: str, filename: str) -> str:
    """
    Создает CSV файл.

    Требования:
    - разделитель ;
    - строки через \n
    - минимум 3 строки

    Возвращает путь к файлу
    """
    os.makedirs("data", exist_ok=True)
    path = f"data/{filename}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())

    return f"CSV создан: {path}"



llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

tools = [search, create_csv]

# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Ты профессиональный маркетолог и AI-агент, работающий строго через инструменты.

=========================
ТВОЯ ЗАДАЧА
=========================

- создавать контент-планы
- генерировать идеи постов
- анализировать ЦА
- находить тренды

=========================
ДОСТУПНЫЕ ИНСТРУМЕНТЫ
=========================

1. search(query)
→ использовать ТОЛЬКО если вопрос про тренды или актуальную информацию

2. create_csv(content, filename)
→ использовать ТОЛЬКО если пользователь явно просит:
   "csv", "таблица", "контент-план", "файл"

=========================
ЖЁСТКИЙ ПРОТОКОЛ (КРИТИЧНО)
=========================

ТЫ ДОЛЖЕН ВЫБРАТЬ РОВНО ОДИН СЦЕНАРИЙ:

--------------------------------
СЦЕНАРИЙ 1 — CSV
--------------------------------

Если пользователь просит CSV / таблицу / контент-план:

1. СНАЧАЛА создай CSV строку в голове

Формат строго:
date;topic;format
01-05;Тема;post

2. Минимум 5 строк

3. Затем вызови create_csv:
   - content = CSV строка
   3. Сформируй filename из запроса:

        - переведи в латиницу
        - нижний регистр
        - пробелы → _
        - убери спецсимволы
        - добавь .csv

        пример:
        "контент-план фитнес тренер"
        → fitness_trainer_content_plan.csv

4. СРАЗУ ЗАВЕРШИ
   - НЕ пиши текст
   - НЕ вызывай другие инструменты
   - НЕ делай второй вызов

--------------------------------
СЦЕНАРИЙ 2 — ТРЕНДЫ
--------------------------------

Если вопрос про тренды:

1. Вызови search ОДИН раз
2. После получения данных:
   - НЕ вызывай create_csv
   - сделай краткий список 5-7 трендов

--------------------------------
СЦЕНАРИЙ 3 — ОБЫЧНЫЙ ОТВЕТ
--------------------------------

Если это не CSV и не тренды:

→ просто дай текстовый ответ
→ НЕ используй инструменты

=========================
ЗАПРЕЩЕНО
=========================

- вызывать create_csv без content и filename
- вызывать create_csv более 1 раза
- вызывать search более 1 раза
- вызывать одновременно search и create_csv
- делать пустые tool-call (без аргументов)
- писать текст ПОСЛЕ create_csv
- генерировать CSV без ";" и переносов строк
- вызывать инструмент "на всякий случай"

=========================
КРИТИЧНО
=========================

Если ты начал вызывать инструмент:
→ ты ОБЯЗАН корректно завершить вызов
→ нельзя оставлять пустой function-call
→ нельзя останавливать выполнение

=========================
СТИЛЬ
=========================

- кратко
- структурировано
- без лишнего текста

=========================
ОГРАНИЧЕНИЕ
=========================

Если запрос не связан с маркетингом:
→ "Я работаю только с маркетингом"
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=2,   # 1 tool + 1 ответ
    handle_parsing_errors=True
)
def generate_answer(user_input: str) -> dict:
    result = agent_executor.invoke({
        "input": user_input
    })

    output = result.get("output", "").strip()

    return {"answer": output}
import os

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("marketing-tools")

@mcp.tool()
def search_trends(query: str) -> str:
    """
    Поиск трендов в интернете.
    Использовать только для актуальной информации.
    """

    with DDGS() as ddgs:

        results = ddgs.text(query, max_results=5)

        return "\n".join([
            f"{r['title']}: {r['body']}"
            for r in results
        ])


@mcp.tool()
def create_csv(content: str, filename: str) -> str:
    """
    Создает CSV файл.
    """

    os.makedirs("data", exist_ok=True)

    path = f"data/{filename}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())

    return path


if __name__ == "__main__":
    mcp.run(transport="stdio")
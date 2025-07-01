import asyncio
from source.backend.app import QueryInput, rag_search_impl
from source.backend.interaction import clean_html_to_one_line


async def main():
    request = QueryInput(query='How to install Git on Linux?')
    response = await rag_search_impl(request)
    cleaned_answer = clean_html_to_one_line(response.answer)

    print(cleaned_answer)

if __name__ == '__main__':
    asyncio.run(main())
    print('!')
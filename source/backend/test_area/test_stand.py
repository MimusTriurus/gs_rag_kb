from source.backend.interaction import answer_question, clean_html_to_one_line, OllamaChatSession, system_prompt
from source.backend.llm_refiners import QueryRefinerBasedOnHistory, QueryRefiner, QueryVariantsRefiner, QueryChecker
from source.backend.settings import LLM_MODEL, OLLAMA_BASE_URL

context = '''
## Pull request: 11215
Close date: 2025-05-19 11:49:27
Version: v0.42.1.3148-ded852e9
## Description
* Nothing player facing.
R3-67981 EAC Deployments
## Changes
* Added config patch for EAC stage environment
## Player Description
* Nothing player facing.
## QA Description
* I've modified the WGC and STEAM config patch to point to prod EAC deploymentID and the rest of the configs to stage EAC deploymentID

## Pull request: 11208
Close date: 2025-05-21 15:33:28
Version: v0.42.1.3151-614ebe63
## Description
N/A: The issue appeared during v42 and shouldn't have reached the players
R3-67865 Removal of the skins unavailable in the Shipping from the bots' config.
## Changes
Hotfix version of
https://stash.wargaming.net/projects/R3/repos/game/pull-requests/11203/overview
* R3-66930 Removal of the skins unavailable in the Shipping from the bots' config.
* Improved the logging across the scenario.
* The code quality improvements in the related files
## Player Description
N/A: The issue appeared during v42 and shouldn't have reached the players
## QA Description
* Note: the original PR contains code style discussion not completely resolved by this moment. But it doesn't affect the code stability and quite minor in terms of code quality, so it doesn't stop, with @g_booker 's approval, this PR from being processed

## Pull request: 11234
Close date: 2025-05-22 09:15:39
Version: v0.42.1.3153-122948c8
## Description
* The game is back to 100% loc
* Some russian wording got fixed to be more idiotmatics
R3-68287 update all languages to 100%
## Changes
* Updated all languages and fixed some suboptimal russian translations
## Player Description
* The game is back to 100% loc
* Some russian wording got fixed to be more idiotmatics
## QA Description
* Amber task https://jira.wargaming.net/browse/R3-67988
* If release QA is faster, they are free to go ahead
'''

topics = [
    'Release notes. Pull request. PR. QA Description. Player Description. Description',
    'Git. Installation. Instructions. Version control. Merge. Conflicts',
    'Weather forcast. London. Video. Sun'
]

questions = [
    'What is the version of PR 11234',
    'How to install Git on Linux',
    'When was this PR closed',
    'Provide me information about Changes for this PR',
    'Who is president Putin?'
]

test_session = OllamaChatSession(LLM_MODEL, system_prompt)

def test_refiner_based_on_history():
    refiner = QueryRefinerBasedOnHistory(model=LLM_MODEL)
    messages = [
        {"role": "user", "content": 'What is the version of PR 11208?'},
        {"role": "assistant", "content": 'v0.42.1.3151-614ebe63'},
        {"role": "user", "content": 'When was this PR closed?'},
        {"role": "assistant", "content": '2025-05-21 15:33:28'},
    ]
    for q in questions:
        new_query = refiner.refine(q, history=messages, tags=[])
        print(new_query)


def test_chat():
    for q in questions:
        answer = answer_question(context, q)
        print(f'==> {clean_html_to_one_line(answer)}\n\n')


def test_refiner_based_on_topics():
    refiner = QueryRefiner(model=LLM_MODEL)
    for q in questions:
        refined_q = refiner.refine(q)
        print(f'--> base question: {q}')
        answer = answer_question(context, q)
        print(f'==> base answer:  {clean_html_to_one_line(answer)}')

        print(f'--> refined question: {refined_q}')
        answer = test_session.ask(context, refined_q)
        print(f'==> refined answer:  {clean_html_to_one_line(answer)}')
        print('')


def test_refiner_if_answer_not_found():
    refiner = QueryVariantsRefiner(model=LLM_MODEL, metadata_topics=topics)
    for q in questions:
        topic, score = refiner.is_relevant_embedding(q)
        if topic and score:
            print(f'{q} [score]: {score} {topic}')
        else:
            print(f'{q} unknown!')
        continue
        if is_relevant:
            refined_queries = refiner.generate_clarifications(q)
            print(f"Clarification variants for: {q}")
            for rq in refined_queries:
                print("-", rq)
        else:
            print(f"The request {q} is not relevant.")
        print()

def is_query_specific():
    queries = [
        'Give me a version for this PR',
        'Give me a version for PR 1234'
    ]
    qc = QueryChecker(LLM_MODEL)
    for q in queries:
        r = qc.is_relevant(q)
        print(r)
    print('=============')



if __name__ == '__main__':
    is_query_specific()
    exit(0)

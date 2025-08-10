def find_overlap(s1, s2):
    """
    Находит максимальное перекрытие между концом s1 и началом s2
    Возвращает длину перекрытия
    """
    max_overlap = 0
    min_len = min(len(s1), len(s2))

    # Проверяем все возможные перекрытия
    for i in range(1, min_len + 1):
        if s1[-i:] == s2[:i]:
            max_overlap = i

    return max_overlap


def merge_strings(strings):
    """
    Объединяет список строк с учетом перекрытий
    """
    if not strings:
        return ""

    # Начинаем с первой строки
    result = strings[0]

    for i in range(1, len(strings)):
        current_string = strings[i]

        # Ищем перекрытие между результатом и текущей строкой
        overlap_len = find_overlap(result, current_string)

        if overlap_len > 0:
            # Добавляем только неперекрывающуюся часть
            result += current_string[overlap_len:]
        else:
            # Если перекрытия нет, проверяем, не содержится ли текущая строка уже в результате
            if current_string not in result:
                result += current_string

    return result


def merge_advanced(strings):
    """
    Продвинутый алгоритм объединения с учетом различных порядков строк
    """
    if not strings:
        return ""

    # Попробуем все возможные порядки и выберем самый короткий результат
    from itertools import permutations

    best_result = None
    min_length = float('inf')

    # Для небольшого количества строк проверим все перестановки
    if len(strings) <= 4:  # Ограничиваем количество перестановок
        for perm in permutations(strings):
            result = merge_strings(list(perm))
            if len(result) < min_length:
                min_length = len(result)
                best_result = result
    else:
        # Для большого количества строк используем жадный подход
        best_result = merge_strings(strings)

    return best_result


# Пример использования с вашими данными
str1 = '''
## 2
Close date: 2025-05-21 15:33:28
Version: v0.42.1.3151-614ebe63
Pull request: PR#11208
## Description
N/A: The issue appeared during v42 and shouldn't have reached the players
R3-67865 Removal of the skins unavailable in the Shipping from the bots' config.
## Changes
Hotfix version of
https://stash.wargaming.net/projects/R3/repos/game/pull-requests/11203/overview
R3-66930 Removal of the skins unavailable in the Shipping from the bots' config.
Improved the logging across the scenario.
## 3
Close date: 2025-05-22 09:15:39
Version: v0.42.1.3153-122948c8
Pull request: PR#11234
## Description
The game is back to 100% loc
Some russian wording got fixed to be more idiotmatics
R3-68287 update all languages to 100%
## Changes
Updated all languages and fixed some suboptimal russian translations
## Player Description
The game is back to 100% loc
Some russian wording got fixed to be more idiotmatics
## QA Description
Amber task https://jira.wargaming.net/browse/R3-67988
'''

str2 = '''
## 2
Close date: 2025-05-21 15:33:28
Version: v0.42.1.3151-614ebe63
Pull request: PR#11208
## Description
N/A: The issue appeared during v42 and shouldn't have reached the players
R3-67865 Removal of the skins unavailable in the Shipping from the bots' config.
## Changes
Hotfix version of
https://stash.wargaming.net/projects/R3/repos/game/pull-requests/11203/overview
R3-66930 Removal of the skins unavailable in the Shipping from the bots' config.
Improved the logging across the scenario.
## 3
Close date: 2025-05-22 09:15:39
Version: v0.42.1.3153-122948c8
Pull request: PR#11234
## Description
The game is back to 100% loc
Some russian wording got fixed to be more idiotmatics
R3-68287 update all languages to 100%
## Changes
Updated all languages and fixed some suboptimal russian translations
## Player Description
The game is back to 100% loc
Some russian wording got fixed to be more idiotmatics
## QA Description
Amber task https://jira.wargaming.net/browse/R3-67988
## 1
Close date: 2025-05-19 11:49:27
Version: v0.42.1.3148-ded852e9
Pull request: PR#11215
## Description
Nothing player facing.
R3-67981 EAC Deployments
## Changes
Added config patch for EAC stage environment
## Player Description
Nothing player facing.
## QA Description
I've modified the WGC and STEAM config patch to point to prod EAC deploymentID and the rest of the configs to stage EAC deploymentID
'''

str3 = '''
## 4
Close date: 2025-05-27 16:11:43
Version: v0.42.1.3155-c1356d49
Pull request: PR#11272
## Description
NO description
R3-68542 - v0.42.1 Branch Preparation
## Changes
Deleted unnecessary gameparams, only supertest left
Skipped onboarding for supertest params
enabled custom game supertest params
Switched on best settings for RC and STAGE
Deleted skins from gifts
## Player Description
NO description
## QA Description
This is QA Settings for release/v0.42.1 ST
## 5
Close date: 2025-05-27 17:11:43
Version: v0.42.1.3155-c1356d49
Pull request: PR#11273
## Description
NO description
R3-68542 - v0.42.1 Branch Preparation
## Changes
Deleted unnecessary gameparams, only supertest left
Skipped onboarding for supertest params
enabled custom game supertest params
Switched on best settings for RC and STAGE
Deleted skins from gifts
## Player Description
NO description
## QA Description
This is QA Settings for release/v0.42.1 ST
'''

str4 = '''
\n
!!! HELLO WORLD !!!
'''

# Дополнительная функция для работы с несколькими строками
def merge_multiple_strings(*args):
    """
    Удобная функция для объединения произвольного количества строк
    """
    return merge_advanced(list(args))

# Пример использования с несколькими строками
# result = merge_multiple_strings(str1, str2, str3, str4)

if __name__ == '__main__':
    strings_to_merge = [str1, str2, str3, str4]
    merged_result = merge_advanced(strings_to_merge)

    print("Объединенный результат:")
    print("=" * 50)
    print(merged_result)
    print("=" * 50)
    print(f"Длина исходных строк: str1={len(str1)}, str2={len(str2)}")
    print(f"Длина объединенного результата: {len(merged_result)}")
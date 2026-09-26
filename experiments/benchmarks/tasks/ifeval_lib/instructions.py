import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union
from absl import logging
import langdetect
from . import instructions_util

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]
_LANGUAGES = instructions_util.LANGUAGE_CODES
_COMPARISON_RELATION = ("less than", "at least")
_MAX_NUM_SENTENCES = 20
_NUM_PLACEHOLDERS = 4
_NUM_BULLETS = 5
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")
_NUM_HIGHLIGHTED_SECTIONS = 4
_SECTION_SPLITER = ("Section", "SECTION")
_NUM_SECTIONS = 5
_NUM_PARAGRAPHS = 5
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")
_NUM_KEYWORDS = 2
_KEYWORD_FREQUENCY = 3
_LETTER_FREQUENCY = 10
_ALL_CAPITAL_WORD_FREQUENCY = 20
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class Instruction:

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):

    def build_description(self, *, language=None):
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))
        self._description_pattern = (
            "Your ENTIRE response should be in {language} language, no other "
            + "language is allowed."
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        return {"language": self._language}

    def get_instruction_args_keys(self):
        return ["language"]

    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class NumberOfSentences(Instruction):

    def build_description(self, *, num_sentences=None, relation=None):
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)
        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation
        self._description_pattern = (
            "Your response should contain {relation} {num_sentences} sentences."
        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        return ["num_sentences", "relation"]

    def check_following(self, value):
        num_sentences = instructions_util.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):

    def build_description(self, *, num_placeholders=None):
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "The response must contain at least {num_placeholders} placeholders "
            + "represented by square brackets, such as [address]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        return ["num_placeholders"]

    def check_following(self, value):
        placeholders = re.findall("\\[.*?\\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):

    def build_description(self, *, num_bullets=None):
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "Your answer must contain exactly {num_bullets} bullet points. "
            + "Use the markdown bullet points such as:\n"
            + "* This is point 1. \n"
            + "* This is point 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        return ["num_bullets"]

    def check_following(self, value):
        bullet_lists = re.findall("^\\s*\\*[^\\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall("^\\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):

    def build_description(self):
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            "Answer with one of the following options: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class HighlightSectionChecker(Instruction):

    def build_description(self, *, num_highlights=None):
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)
        self._description_pattern = (
            "Highlight at least {num_highlights} sections in your answer with "
            + "markdown, i.e. *highlighted section*."
        )
        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        return ["num_highlights"]

    def check_following(self, value):
        num_highlights = 0
        highlights = re.findall("\\*[^\\n\\*]*\\*", value)
        double_highlights = re.findall("\\*\\*[^\\n\\*]*\\*\\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1
        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):

    def build_description(self, *, section_spliter=None, num_sections=None):
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)
        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)
        self._description_pattern = (
            "Your response must have {num_sections} sections. Mark the beginning "
            + "of each section with {section_spliter} X, such as:\n"
            + "{section_spliter} 1\n"
            + "[content of section 1]\n"
            + "{section_spliter} 2\n"
            + "[content of section 2]"
        )
        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        section_splitter_patten = "\\s?" + self._section_spliter + "\\s?\\d+\\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):

    def build_description(self, *, num_paragraphs=None):
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)
        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. "
            + "Paragraphs are separated with the markdown divider: ***"
        )
        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        return ["num_paragraphs"]

    def check_following(self, value):
        paragraphs = re.split("\\s?\\*\\*\\*\\s?", value)
        num_paragraphs = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False
        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):

    def build_description(self, *, postscript_marker=None):
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)
        self._description_pattern = (
            "At the end of your response, please explicitly add a postscript "
            + "starting with {postscript}"
        )
        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        return ["postscript_marker"]

    def check_following(self, value):
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = "\\s*p\\.\\s?p\\.\\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = "\\s*p\\.\\s?s\\..*$"
        else:
            postscript_pattern = "\\s*" + self._postscript_marker.lower() + ".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class KeywordChecker(Instruction):

    def build_description(self, *, keywords=None):
        if not keywords:
            self._keywords = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)
        self._description_pattern = "Include keywords {keywords} in the response."
        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        return ["keywords"]

    def check_following(self, value):
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()
        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)
        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation
        self._description_pattern = (
            "In your response, the word {keyword} should appear {relation} "
            + "{frequency} times."
        )
        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):

    def build_description(self, *, num_words=None, relation=None):
        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )
        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation
        self._description_pattern = "Answer with {relation} {num_words} words."
        return self._description_pattern.format(
            relation=self._comparison_relation, num_words=self._num_words
        )

    def get_instruction_args(self):
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        return ["num_words", "relation"]

    def check_following(self, value):
        num_words = instructions_util.count_words(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words


class JsonFormat(Instruction):

    def build_description(self):
        self._description_pattern = "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)
        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)
        self._first_word = first_word
        if self._first_word is None:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()
        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. "
            + "Paragraphs and only paragraphs are separated with each other by two "
            + "new lines as if it was '\\n\\n' in python. "
            + "Paragraph {nth_paragraph} must start with word {first_word}."
        )
        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        paragraphs = re.split("\\n\\n", value)
        num_paragraphs = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False
        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}
        word = paragraph.split()[0].strip()
        word = word.lstrip("'")
        word = word.lstrip('"')
        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()
        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class ForbiddenWords(Instruction):

    def build_description(self, forbidden_words=None):
        if not forbidden_words:
            self._forbidden_words = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            "Do not include keywords {forbidden_words} in the response."
        )
        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        return ["forbidden_words"]

    def check_following(self, value):
        for word in self._forbidden_words:
            if re.search("\\b" + word + "\\b", value, flags=re.IGNORECASE):
                return False
        return True


class TwoResponsesChecker(Instruction):

    def build_description(self):
        self._description_pattern = "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):

    def build_description(self, *, prompt_to_repeat=None):
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = "First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)"
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class EndChecker(Instruction):

    def build_description(self, *, end_phrase=None):
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = "Finish your response with this exact phrase {ender}. No other words should follow this phrase."
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        return ["end_phrase"]

    def check_following(self, value):
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):

    def build_description(self):
        self._description_pattern = "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        pattern = "<<[^\\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(Instruction):

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        if (
            not letter
            or len(letter) > 1
            or ord(letter.lower()) < 97
            or (ord(letter.lower()) > 122)
        ):
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()
        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)
        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation
        self._description_pattern = "In your response, the letter {letter} should appear {let_relation} {let_frequency} times."
        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        value = value.lower()
        letters = collections.Counter(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):

    def build_description(self):
        self._description_pattern = (
            "Your entire response should be in English, and in all capital letters."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class LowercaseLettersEnglishChecker(Instruction):

    def build_description(self):
        self._description_pattern = "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class CommaChecker(Instruction):

    def build_description(self):
        self._description_pattern = (
            "In your entire response, refrain from the use of any commas."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return not re.search("\\,", value)


class CapitalWordFrequencyChecker(Instruction):

    def build_description(self, capital_frequency=None, capital_relation=None):
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)
        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {capital_relation} is given."
            )
        self._description_pattern = "In your response, words with all capital letters should appear {relation} {frequency} times."
        return self._description_pattern.format(
            frequency=self._frequency, relation=self._comparison_relation
        )

    def get_instruction_args(self):
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        words = instructions_util.nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]
        capital_words = len(capital_words)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(Instruction):

    def build_description(self):
        self._description_pattern = (
            "Wrap your entire response with double quotation marks."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and (value[-1] == '"')

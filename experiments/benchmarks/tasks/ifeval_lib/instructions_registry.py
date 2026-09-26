from . import instructions

_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"
INSTRUCTION_DICT = {
    _KEYWORD + "existence": instructions.KeywordChecker,
    _KEYWORD + "frequency": instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": instructions.ParagraphChecker,
    _LENGTH + "number_words": instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": instructions.PlaceholderChecker,
    _CONTENT + "postscript": instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": instructions.BulletListChecker,
    _FORMAT + "constrained_response": instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": instructions.HighlightSectionChecker,
    _FORMAT + "multiple_sections": instructions.SectionChecker,
    _FORMAT + "json_format": instructions.JsonFormat,
    _FORMAT + "title": instructions.TitleChecker,
    _COMBINATION + "two_responses": instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": instructions.EndChecker,
    _CHANGE_CASES + "capital_word_frequency": instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": instructions.CommaChecker,
    _STARTEND + "quotation": instructions.QuotationChecker,
}

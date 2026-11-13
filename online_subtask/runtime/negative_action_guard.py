"""Recognize a small set of direct negative action constraints without edits.

True means only that the action token has a supported, immediately adjacent
negative prefix. False also covers unknown or ambiguous syntax. This helper is
not an English semantic parser and does not certify a public plan's accuracy.
Callers must pass the action's actual start in the same, unmodified text.
"""
import re


# Keep this allowlist finite: intervening verbs, objects, conjunctions, and
# modifiers such as "only" or "necessarily" must not extend negative scope.
_ADVERBS = (
    'accidentally', 'actually', 'deliberately', 'directly', 'ever', 'fully',
    'gently', 'immediately', 'intentionally', 'prematurely', 'slightly',
)
_ADVERB_PATTERN = '(?:' + '|'.join(_ADVERBS) + ')'
_PREFIX_PATTERN = (
    r'(?:without|avoid|avoiding|never|cannot|'
    r'(?:do|does|did|must|should|can|could|will|would|shall)[ \t]+not|'
    r'(?:don|doesn|didn|mustn|shouldn|can|couldn|won|wouldn|shan)[\'’]t)'
)
_DIRECT_PREFIX = re.compile(
    r'\b' + _PREFIX_PATTERN
    + r'(?:[ \t]+' + _ADVERB_PATTERN + r'){0,2}[ \t]+$',
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r'[.;:!?\n\r]|\b(?:and|but|then|or|yet|however)\b', re.IGNORECASE,
)
# A preceding second negation or scope-changing verb makes even a direct
# suffix ambiguous ("do not avoid releasing", "fail to avoid releasing").
# This veto never extends a negative prefix across intervening words.
_AMBIGUOUS_SCOPE = re.compile(
    r'\b(?:not|no|never|without|avoid|avoiding|cannot|'
    r'(?:don|doesn|didn|mustn|shouldn|can|couldn|won|wouldn|shan|isn|aren|wasn|weren)[\'’]t|'
    r'fail|fails|failed|failing|stop|stops|stopped|stopping|cease|ceases|ceased|ceasing)\b',
    re.IGNORECASE,
)


def _has_unparsed_quote(text):
    """Reject quote syntax; internal word apostrophes are not quote markers."""
    for index, character in enumerate(text):
        if character in {'"', '“', '”', '‘', '`'}:
            return True
        if character in {"'", '’'}:
            if not (0 < index < len(text) - 1
                    and text[index - 1].isalnum() and text[index + 1].isalnum()):
                return True
    return False


def is_negated(text: str, action_start_index: int) -> bool:
    """Return True only for an unambiguous supported direct negative prefix.

    Supported prefixes are without/avoid/avoiding/never, cannot, listed
    auxiliary + not phrases, and their listed contractions. At most two
    allowlisted adverbs may intervene. Quotes, other negative scope in the
    same clause, punctuation, arbitrary words, and malformed indices return
    False. No part of the supplied text is changed or returned as a rewrite.
    """
    if (type(text) is not str or type(action_start_index) is not int
            or not 0 <= action_start_index < len(text)):
        return False
    if not ('A' <= text[action_start_index] <= 'Z'
            or 'a' <= text[action_start_index] <= 'z'):
        return False
    if (action_start_index and
            (text[action_start_index - 1].isalnum() or text[action_start_index - 1] == '_')):
        return False
    if _has_unparsed_quote(text):
        return False
    before_action = text[:action_start_index]
    match = _DIRECT_PREFIX.search(before_action)
    if match is None:
        return False
    before_prefix = before_action[:match.start()]
    boundaries = list(_CLAUSE_BOUNDARY.finditer(before_prefix))
    clause_start = boundaries[-1].end() if boundaries else 0
    if _AMBIGUOUS_SCOPE.search(before_prefix[clause_start:]):
        return False
    return True

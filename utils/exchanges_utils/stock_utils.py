import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Iterable

try:
    from pypinyin import Style, lazy_pinyin, pinyin as pypinyin_pinyin
except ImportError:
    Style = None
    lazy_pinyin = None
    pypinyin_pinyin = None

from env_setting import ROOT

code_with_exchanges = json.load(open(f"{ROOT}/utils/exchanges_meta/code_with_exchanges.json", "r"))
stock_code_to_name_map = json.load(open(f"{ROOT}/utils/exchanges_meta/stock_code_to_name.json", "r", encoding="utf-8"))
stock_name_to_code_map = {v: k for k, v in stock_code_to_name_map.items()}
code_list = json.load(open(f"{ROOT}/utils/exchanges_meta/code_list_append.json", "r"))

MAIN_BOARD_PREFIXES = ("00", "60")
CHINEXT_PREFIXES = ("30",)
STAR_BOARD_PREFIXES = ("68",)
BSE_PREFIXES = ("43", "83", "87", "88")


@dataclass(frozen=True)
class StockSearchRecord:
    code: str
    bare_code: str
    name: str
    pinyin: str
    initials: str
    pinyin_all: frozenset[str] = field(default_factory=frozenset)
    initials_all: frozenset[str] = field(default_factory=frozenset)

    @property
    def display_text(self) -> str:
        return f"{self.code}  {self.name}"


def get_code_with_exchange(code):
    return code_with_exchanges.get(code, None)

def get_code_list():
    return code_list

def get_code_idx(code):
    if not code:
        return -1
    code_with_exchange = get_code_with_exchange(code)
    if code_with_exchange is not None:
        code = code_with_exchange
    try:
        return code_list.index(code)
    except ValueError:
        return -1

def normalize_stock_code(code: str | None) -> str:
    text = str(code or "").strip().lower()
    if not text:
        return ""
    if text in stock_code_to_name_map:
        return text
    bare_code = extract_bare_stock_code(text)
    return get_code_with_exchange(bare_code) or text


def extract_bare_stock_code(code: str | None) -> str:
    text = str(code or "").strip().lower()
    if not text:
        return ""
    for sep in (".", "_"):
        if sep in text:
            parts = [part for part in text.split(sep) if part]
            digit_part = next((part for part in parts if part.isdigit()), "")
            if digit_part:
                return digit_part
    return text


def stock_code_to_name(code: str) -> str | None:
    full_code = normalize_stock_code(code)
    name = stock_code_to_name_map.get(full_code)
    return name


def stock_name_to_code(name: str) -> str | list | None:
    code = stock_name_to_code_map.get(name)
    return code


def get_all_codes(bare: bool = False) -> list[str]:
    """返回全部股票代码列表。bare=True 时返回不带交易所前缀的 6 位纯数字代码。"""
    if bare:
        return [extract_bare_stock_code(code) for code in stock_code_to_name_map]
    return list(stock_code_to_name_map)


def list_stock_search_records(allowed_codes: Iterable[str] | None = None) -> list[StockSearchRecord]:
    records = list(_all_stock_search_records())
    if allowed_codes is None:
        return records

    normalized_allowed = {normalize_stock_code(code) for code in allowed_codes if str(code or "").strip()}
    filtered = [record for record in records if record.code in normalized_allowed]

    missing_codes = sorted(normalized_allowed.difference({record.code for record in filtered}))
    for code in missing_codes:
        if not code:
            continue
        filtered.append(
            StockSearchRecord(
                code=code,
                bare_code=extract_bare_stock_code(code),
                name=stock_code_to_name_map.get(code, code),
                pinyin="",
                initials="",
                pinyin_all=frozenset(),
                initials_all=frozenset(),
            )
        )
    return filtered


def search_stock_records(
    query: str,
    allowed_codes: Iterable[str] | None = None,
    limit: int = 80,
) -> list[StockSearchRecord]:
    records = list_stock_search_records(allowed_codes=allowed_codes)
    return filter_stock_search_records(query=query, records=records, limit=limit)


def filter_stock_search_records(
    query: str,
    records: Iterable[StockSearchRecord],
    limit: int = 80,
) -> list[StockSearchRecord]:
    records = list(records)
    if limit <= 0:
        return []

    query_text = str(query or "").strip().lower()
    if not query_text:
        return records[:limit]

    ranked: list[tuple[tuple[int, int, str], StockSearchRecord]] = []
    for record in records:
        match_key = _stock_record_match_key(record, query_text)
        if match_key is None:
            continue
        ranked.append((match_key, record))

    ranked.sort(key=lambda item: item[0])
    return [record for _, record in ranked[:limit]]


def resolve_stock_query(query: str, allowed_codes: Iterable[str] | None = None) -> StockSearchRecord | None:
    query_text = str(query or "").strip()
    if not query_text:
        return None

    matches = search_stock_records(query_text, allowed_codes=allowed_codes, limit=20)
    if not matches:
        return None

    lowered = query_text.lower()
    for record in matches:
        if lowered in {record.code, record.bare_code, record.name.lower(), record.initials, record.pinyin}:
            return record
    if len(matches) == 1:
        return matches[0]
    return None


@lru_cache(maxsize=1)
def _all_stock_search_records() -> tuple[StockSearchRecord, ...]:
    records: list[StockSearchRecord] = []
    for raw_code, raw_name in sorted(stock_code_to_name_map.items()):
        code = normalize_stock_code(raw_code)
        name = str(raw_name or "").strip()
        pinyin, initials, pinyin_all, initials_all = _name_index_tokens(name)
        records.append(
            StockSearchRecord(
                code=code,
                bare_code=extract_bare_stock_code(code),
                name=name,
                pinyin=pinyin,
                initials=initials,
                pinyin_all=pinyin_all,
                initials_all=initials_all,
            )
        )
    return tuple(records)


def _name_index_tokens(name: str) -> tuple[str, str, frozenset[str], frozenset[str]]:
    """Return (primary_pinyin, initials, pinyin_all, initials_all) for *name*."""
    name_text = str(name or "").strip()
    if not name_text:
        return "", "", frozenset(), frozenset()
    if lazy_pinyin is None or Style is None:
        ascii_text = re.sub(r"[^a-z0-9]+", "", name_text.lower())
        return ascii_text, ascii_text, frozenset({ascii_text}), frozenset({ascii_text})

    pinyin_list = lazy_pinyin(name_text, errors="ignore")
    initial_list = lazy_pinyin(name_text, style=Style.FIRST_LETTER, errors="ignore")
    pinyin = "".join(part.strip().lower() for part in pinyin_list if str(part).strip())
    initials = "".join(part.strip().lower() for part in initial_list if str(part).strip())

    # Build all possible pinyin readings via cartesian product of
    # every character's full reading list (handles multi-pronunciation).
    pinyin_all: set[str] = set()
    initials_all: set[str] = set()
    if pypinyin_pinyin is not None and Style is not None:
        try:
            all_readings = pypinyin_pinyin(name_text, style=Style.NORMAL, errors="ignore", heteronym=True)
            # Each element is a list like ['chang', 'zhang'] for 长.
            per_char: list[list[str]] = []
            for readings in all_readings:
                cleaned = [r.strip().lower() for r in readings if r and r.strip()]
                if cleaned:
                    per_char.append(cleaned)
            if per_char:
                # Cartesian product — cap at 64 combos to avoid explosion.
                for combo in product(*per_char):
                    variant = "".join(combo)
                    pinyin_all.add(variant)
                    initials_all.add("".join(c[0] for c in combo))
                    if len(pinyin_all) >= 64:
                        break
        except Exception:
            pass
    if not pinyin_all:
        pinyin_all = {pinyin}
    if not initials_all:
        initials_all = {initials}

    return pinyin, initials, frozenset(pinyin_all), frozenset(initials_all)


def _stock_record_match_key(record: StockSearchRecord, query: str) -> tuple[int, int, str] | None:
    name_lower = record.name.lower()
    exact_tokens = {record.code, record.bare_code, name_lower, record.initials, record.pinyin}
    if query in exact_tokens:
        return (0, len(record.code), record.code)

    if record.code.startswith(query) or record.bare_code.startswith(query):
        return (1, len(record.bare_code), record.code)
    if record.initials.startswith(query) or any(v.startswith(query) for v in record.initials_all):
        return (2, len(record.initials), record.code)
    if record.pinyin.startswith(query) or any(v.startswith(query) for v in record.pinyin_all):
        return (3, len(record.pinyin), record.code)
    if name_lower.startswith(query):
        return (4, len(record.name), record.code)
    if query in record.code or query in record.bare_code:
        return (5, record.bare_code.find(query), record.code)
    if query in record.initials or any(query in v for v in record.initials_all):
        return (6, record.initials.find(query), record.code)
    if query in record.pinyin or any(query in v for v in record.pinyin_all):
        return (7, record.pinyin.find(query), record.code)
    if query in name_lower:
        return (8, name_lower.find(query), record.code)
    return None


def price_limit_status(close, preclose, limit=0.1):
    up_limit = round(preclose * (1 + limit), 2)
    down_limit = round(preclose * (1 - limit), 2)
    if close >= up_limit:
        return "UP_LIMIT"
    elif close <= down_limit:
        return "DOWN_LIMIT"
    else:
        return "NORMAL"

def get_price_limit(code, isST, isNew):
    """
    获取A股股票的涨跌停限制幅度

    参数:
        code: 股票代码（不带交易所前后缀），字符串
        isST: 是否为ST股票，布尔值
        isNew: 是否为新股（上市首日），布尔值

    返回:
        涨跌停限制百分比（如0.10表示10%），如果新股上市首日无涨跌幅限制则返回0
    """
    # 确保code是字符串
    code = str(code).strip()
    if '.' in code:
        code = code.split('.')[1]

    # 新股上市首日处理
    if isNew:
        # 主板新股上市首日：44%
        if code.startswith(MAIN_BOARD_PREFIXES):
            return 0.44
        # 创业板新股上市首日：44%
        elif code.startswith(CHINEXT_PREFIXES):
            return 0.44
        # 科创板新股上市首日：44%
        elif code.startswith(STAR_BOARD_PREFIXES):
            return 0.44
        # 北交所新股上市首日：无涨跌幅限制
        elif code.startswith(BSE_PREFIXES):
            return 0  # 无涨跌幅限制
        else:
            return -1  # 未知板块

    # 非新股处理
    # 主板（沪市主板、深市主板、中小板）
    if code.startswith(MAIN_BOARD_PREFIXES):
        if isST:
            return 0.05  # ST股票5%限制
        else:
            return 0.10  # 普通股票10%限制

    # 创业板
    elif code.startswith(CHINEXT_PREFIXES):
        # 创业板ST股票也是20%限制
        return 0.20

    # 科创板
    elif code.startswith(STAR_BOARD_PREFIXES):
        # 科创板股票都是20%限制（科创板没有ST，但有风险警示）
        return 0.20

    # 北交所
    elif code.startswith(BSE_PREFIXES):
        # 北交所股票都是30%限制
        return 0.30

    # 未知板块
    else:
        return -1


if __name__ == "__main__":
    print(get_code_idx("000001"))  # 测试获取股票索引
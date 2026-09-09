"""
在线歌词检索
用 ID3 里的歌名/歌手到网易云音乐检索官方歌词，按时长与标题/歌手相似度挑选最贴合的版本
"""
import re
from difflib import SequenceMatcher

import requests

from engine.lrc_parser import parse_lrc, has_usable_lyrics

SEARCH_URL = 'https://music.163.com/api/search/get/web'
LYRIC_URL = 'https://music.163.com/api/song/lyric'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://music.163.com/',
    'Accept': 'application/json, text/plain, */*',
}

TIMEOUT = 8
MAX_LYRIC_FETCHES = 4
SEARCH_LIMIT = 15

# 翻唱/改编标记，比对标题时先剥掉，但保留原文用于歌手匹配
BRACKET_RE = re.compile(r'[（(\[【][^）)\]】]*[）)\]】]')
VERSION_WORDS_RE = re.compile(
    r'\b(dj|remix|cover|live|acoustic|instrumental|demo|mix|edit|version)\b',
    re.IGNORECASE,
)
VERSION_SUFFIXES = (
    '伴奏', '纯音乐', '童声版', '烟嗓版', '雷鬼版', '新版', '旧版', '完整版',
    '片段', '剪辑版', '加速版', '减速版', '翻唱', '男声版', '女声版', '合唱版',
)


def _norm_title(text):
    if not text:
        return ''
    cleaned = BRACKET_RE.sub(' ', text)
    cleaned = VERSION_WORDS_RE.sub(' ', cleaned)
    for suffix in VERSION_SUFFIXES:
        cleaned = cleaned.replace(suffix, ' ')
    cleaned = re.sub(r'[\s\-_·、,，.。:：/|!！?？"\'""''()（）\[\]【】]+', '', cleaned)
    return cleaned.lower()


def _norm_artist(text):
    if not text:
        return ''
    return re.sub(r'[\s\-_·、,，.。:：/|&]+', '', text).lower()


def _search(session, keyword):
    resp = session.get(
        SEARCH_URL,
        params={'type': 1, 's': keyword, 'limit': SEARCH_LIMIT, 'offset': 0},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    songs = ((payload.get('result') or {}).get('songs')) or []

    candidates = []
    for song in songs:
        song_id = song.get('id')
        name = song.get('name') or ''
        if not song_id or not name:
            continue
        artists = '/'.join(
            a.get('name') or '' for a in (song.get('artists') or []) if a.get('name')
        )
        duration = (song.get('duration') or 0) / 1000.0
        candidates.append({
            'id': song_id,
            'name': name,
            'artists': artists,
            'duration': duration,
        })
    return candidates


def _score(candidate, title, artist, duration):
    want_title = _norm_title(title)
    cand_title = _norm_title(candidate['name'])
    if not want_title or not cand_title:
        return -1.0

    title_sim = SequenceMatcher(None, want_title, cand_title).ratio()
    if title_sim < 0.55:
        return -1.0

    score = title_sim * 1.2

    want_artist = _norm_artist(artist)
    if want_artist:
        cand_artist = _norm_artist(candidate['artists'])
        # 歌手名经常写在标题里，例如「万海东 山风山风等等我」「... (Cover 万海东)」
        haystack = cand_artist + _norm_artist(candidate['name'])
        if want_artist and want_artist in haystack:
            score += 0.8
        else:
            score += SequenceMatcher(None, want_artist, cand_artist).ratio() * 0.4

    if duration and candidate['duration']:
        drift = abs(candidate['duration'] - duration) / max(duration, 1.0)
        score -= min(drift, 1.0)

    return score


def _fetch_lyric(session, song_id):
    resp = session.get(
        LYRIC_URL,
        params={'id': song_id, 'lv': -1, 'kv': -1, 'tv': -1, 'os': 'pc'},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get('code') not in (200, None):
        return ''
    if payload.get('pureMusic'):
        return ''
    return ((payload.get('lrc') or {}).get('lyric')) or ''


def fetch_lyrics(title, artist=None, duration=None):
    """
    检索在线歌词

    返回: {'lines': [...], 'synced': bool, 'matched': {...}} 或 None（离线/未命中）
    """
    title = (title or '').strip()
    if not title:
        return None

    queries = [f'{title} {artist}'.strip()]
    if artist:
        queries.append(title)

    seen = set()
    candidates = []
    try:
        session = requests.Session()
        session.headers.update(HEADERS)

        for keyword in queries:
            for cand in _search(session, keyword):
                if cand['id'] in seen:
                    continue
                seen.add(cand['id'])
                candidates.append(cand)

        if not candidates:
            print(f'[LyricsProvider] No candidates for {title!r}')
            return None

        candidates.sort(
            key=lambda c: _score(c, title, artist, duration), reverse=True
        )
        candidates = [
            c for c in candidates if _score(c, title, artist, duration) > 0
        ][:MAX_LYRIC_FETCHES]

        for cand in candidates:
            text = _fetch_lyric(session, cand['id'])
            parsed = parse_lrc(text)
            if not has_usable_lyrics(parsed):
                print(f"[LyricsProvider] id={cand['id']} {cand['name']} -> 无可用歌词")
                continue

            print(
                f"[LyricsProvider] Matched id={cand['id']} "
                f"{cand['name']} - {cand['artists']} "
                f"({cand['duration']:.1f}s, {len(parsed['lines'])} lines, "
                f"synced={parsed['synced']})"
            )
            return {
                'lines': parsed['lines'],
                'synced': parsed['synced'],
                'matched': cand,
            }

    except requests.RequestException as e:
        print(f'[LyricsProvider] Network unavailable: {type(e).__name__}: {e}')
        return None
    except Exception as e:
        print(f'[LyricsProvider] Unexpected error: {type(e).__name__}: {e}')
        return None

    print(f'[LyricsProvider] No usable lyrics among {len(candidates)} candidates')
    return None

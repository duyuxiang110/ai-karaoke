"""
LRC 歌词文本解析
兼容带时间戳的同步歌词与纯文本的非同步歌词，过滤元数据标签和制作人员字幕
"""
import re

META_TAG_RE = re.compile(r'^\[([a-zA-Z#][a-zA-Z0-9#_]*):(.*)\]$')

# [mm:ss] / [mm:ss.x] / [mm:ss.xx] / [mm:ss.xxx] / [mm:ss.xx-1]
TIMESTAMP_RE = re.compile(r'\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?(?P<marker>-?\d+)?\]')

# 制作人员字幕，必须带冒号才算，避免误杀真实歌词
CREDIT_RE = re.compile(
    r'^\s*(作词|作曲|编曲|制作人|制作|混音|录音|母带|和声|和音|吉他|贝斯|贝司|鼓|弦乐|'
    r'钢琴|小提琴|大提琴|笛|唢呐|统筹|监制|出品|发行|策划|文案|录音棚|配唱|'
    r'词|曲|OP|SP|op|sp)\s*[:：/]'
)

NOISE_TEXTS = {
    '纯音乐，请欣赏', '纯音乐,请欣赏', '纯音乐', '请欣赏',
    '暂无歌词', '没有歌词', '该歌曲暂无歌词', 'instrumental',
}

KNOWN_META_KEYS = {
    'ti', 'ar', 'al', 'by', 'offset', 're', 've', 'hash', 'sign', 'qq',
    'total', 'language', 'au', 'length', 'tool', 'file', 'version', 'id',
}


def _normalize(text):
    return text.replace('\u3000', ' ').strip()


def _is_noise(text):
    stripped = text.strip().strip('。．.!！~～ ')
    if not stripped:
        return True
    if stripped.lower() in NOISE_TEXTS:
        return True
    return bool(CREDIT_RE.match(stripped))


def parse_lrc(text):
    """
    解析 LRC 文本

    返回:
      {
        'lines': [{'text': str, 'start': float|None, 'end': float|None}],
        'synced': bool,      # 是否带有可用的真实时间戳
        'meta': {ti/ar/al/by...},
      }
    """
    if not text:
        return {'lines': [], 'synced': False, 'meta': {}}

    meta = {}
    offset_ms = 0.0
    entries = []          # (start, text)
    plain_texts = []      # 无时间戳的纯文本行
    timed_count = 0
    marker_only_count = 0

    for raw in text.splitlines():
        line = _normalize(raw)
        if not line:
            continue

        # 整行是元数据标签，如 [ti:标题] [ar:歌手] [offset:+500]
        meta_match = META_TAG_RE.match(line)
        if meta_match and not TIMESTAMP_RE.match(line):
            key = meta_match.group(1).lower()
            value = meta_match.group(2).strip()
            if key in KNOWN_META_KEYS:
                meta[key] = value
                if key == 'offset':
                    try:
                        offset_ms = float(value)
                    except ValueError:
                        pass
                continue

        # 收集本行所有时间戳；剩下的部分是歌词文本
        stamps = []
        last_end = 0
        for m in TIMESTAMP_RE.finditer(line):
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            frac = m.group(3) or '0'
            # 小数位可能是 1~3 位，统一按毫秒解释
            millis = float('0.' + frac)
            start = minutes * 60 + seconds + millis
            marker = m.group('marker')
            # 网易云用 [00:00.00-1] 表示该条歌词无有效时间轴
            if marker is not None and marker.strip('-').isdigit() and int(marker) < 0:
                marker_only_count += 1
            else:
                stamps.append(start)
                timed_count += 1
            last_end = m.end()

        body = _normalize(line[last_end:]) if last_end else line

        if stamps:
            for start in stamps:
                if body and not _is_noise(body):
                    entries.append((start, body))
        elif body and not _is_noise(body):
            plain_texts.append(body)

    # offset 为正表示歌词整体提前
    shift = offset_ms / 1000.0

    if entries:
        entries.sort(key=lambda e: e[0])
        starts = [e[0] for e in entries]
        gaps = [b - a for a, b in zip(starts, starts[1:]) if b > a]
        tail = min(max(sum(gaps) / len(gaps), 1.0), 8.0) if gaps else 4.0

        lines = []
        for i, (start, body) in enumerate(entries):
            end = starts[i + 1] if i + 1 < len(starts) else start + tail
            lines.append({
                'text': body,
                'start': round(max(start - shift, 0.0), 3),
                'end': round(max(end - shift, 0.0), 3),
            })
        return {'lines': lines, 'synced': True, 'meta': meta}

    if plain_texts:
        lines = [{'text': t, 'start': None, 'end': None} for t in plain_texts]
        return {'lines': lines, 'synced': False, 'meta': meta}

    # 只有 [xx:xx-1] 之类的无效时间轴条目，视为没有歌词
    return {'lines': [], 'synced': False, 'meta': meta}


def has_usable_lyrics(parsed, min_lines=4):
    """判断解析结果是否值得展示"""
    return bool(parsed) and len(parsed.get('lines', [])) >= min_lines

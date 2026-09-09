"""
歌词对齐引擎
优先级: MP3 ID3 内嵌歌词(USLT) > 同名 .lrc 文件 > 在线检索(网易云) > 无歌词

带时间戳的歌词直接采用官方时间轴；只有拿到无时间戳的纯文本歌词时，
才按人声音轨的有声段落推算时间轴。
"""
import os
import re
import struct

from engine.lrc_parser import parse_lrc, has_usable_lyrics
from engine.lyrics_provider import fetch_lyrics
from engine.vocal_segments import singing_segments, vocal_segments

_TEXT_CODECS = {0: 'latin-1', 1: 'utf-16', 2: 'utf-16-be', 3: 'utf-8'}

_DOWNLOAD_SUFFIX_RE = re.compile(r'[#_]\w{5,}$')
_AUDIO_EXT_RE = re.compile(
    r'(?:\.(?:vocals|instrumental))?\.(?:mp3|wav|flac|m4a|aac|ogg)$', re.IGNORECASE
)
_BRACKET_RE = re.compile(r'[（(\[【][^）)\]】]*[）)\]】]')
_SEPARATOR_RE = re.compile(r'\s*-\s*|\s+–\s+')


def align(vocal_path, original_path=None, lrc_path=None):
    """
    提取歌词及其时间轴

    返回: (lyrics, source)
      lyrics: [{'text', 'start', 'end'}]
      source: {'type': 'id3'|'lrc'|'online'|'none', 'label': str, 'synced': bool, 'reason': str}
    """
    # 1. 用户在客户端手动挂的 .lrc，优先级最高
    if lrc_path:
        parsed = _read_lrc_file(lrc_path)
        if has_usable_lyrics(parsed):
            print(f"[LyricAligner] Manual LRC {lrc_path}: {len(parsed['lines'])} lines")
            return parsed['lines'], {
                'type': 'lrc',
                'label': os.path.basename(lrc_path),
                'synced': parsed['synced'],
                'reason': '',
            }
        print(f'[LyricAligner] Manual LRC unusable: {lrc_path}')

    tags = _read_id3_tags(original_path)

    # 2. MP3 内嵌歌词
    if tags['lyrics_text']:
        parsed = parse_lrc(tags['lyrics_text'])
        if has_usable_lyrics(parsed):
            print(f"[LyricAligner] ID3 USLT: {len(parsed['lines'])} lines")
            return parsed['lines'], {
                'type': 'id3',
                'label': 'MP3 内嵌歌词',
                'synced': parsed['synced'],
                'reason': '',
            }

    # 3. 同名 .lrc 文件
    same_name_lrc = _find_lrc_file(vocal_path, original_path)
    if same_name_lrc:
        parsed = _read_lrc_file(same_name_lrc)
        if has_usable_lyrics(parsed):
            print(f"[LyricAligner] LRC {same_name_lrc}: {len(parsed['lines'])} lines")
            return parsed['lines'], {
                'type': 'lrc',
                'label': os.path.basename(same_name_lrc),
                'synced': parsed['synced'],
                'reason': '',
            }

    # 4. 在线检索
    title = tags['title']
    artist = tags['artist']
    if not title:
        title, artist = _tags_from_filename(original_path or vocal_path)

    if title:
        duration = _audio_duration(vocal_path)
        result = fetch_lyrics(title, artist, duration)
        if result:
            lines = result['lines']
            if not result['synced']:
                lines = _distribute_over_vocals(lines, vocal_path)

            if lines:
                matched = result['matched']
                return lines, {
                    'type': 'online',
                    'label': f"{matched['name']} - {matched['artists']}",
                    'synced': result['synced'],
                    'reason': '',
                }

    return [], {
        'type': 'none',
        'label': '',
        'synced': False,
        'reason': '未找到匹配歌词（离线或曲库无此歌），可在歌曲同目录放一个同名 .lrc 文件',
    }


# ─── ID3 标签 ───

def _decode_text_field(data):
    """解码 ID3 文本帧：encoding(1) + 文本，多值以 \\x00 分隔"""
    if not data:
        return ''
    codec = _TEXT_CODECS.get(data[0], 'utf-8')
    text = data[1:].decode(codec, errors='ignore')
    parts = [p.strip() for p in text.split('\x00') if p.strip()]
    return parts[0] if parts else ''


def _read_id3_tags(mp3_path):
    """
    读取 ID3v2 标签中的标题(TIT2)、歌手(TPE1)、内嵌歌词(USLT)
    """
    tags = {'title': '', 'artist': '', 'lyrics_text': ''}
    if not mp3_path or not os.path.exists(mp3_path):
        return tags

    try:
        with open(mp3_path, 'rb') as f:
            header = f.read(10)
            if len(header) < 10 or header[:3] != b'ID3':
                return tags

            version_major = header[3]
            if version_major < 3:
                # ID3v2.2 用 3 字节帧头，与 v2.3/v2.4 不兼容
                return tags

            size = header[6:10]
            tag_size = (size[0] << 21) | (size[1] << 14) | (size[2] << 7) | size[3]
            tag = f.read(tag_size)

            pos = 0
            while pos + 10 <= len(tag):
                frame_id = tag[pos:pos + 4]
                if frame_id == b'\x00\x00\x00\x00':
                    break  # 后面是填充零

                raw_size = tag[pos + 4:pos + 8]
                if version_major == 4:
                    frame_size = (
                        (raw_size[0] << 21) | (raw_size[1] << 14)
                        | (raw_size[2] << 7) | raw_size[3]
                    )
                else:
                    frame_size = struct.unpack('>I', raw_size)[0]

                if frame_size <= 0 or pos + 10 + frame_size > len(tag):
                    break

                data = tag[pos + 10:pos + 10 + frame_size]
                pos += 10 + frame_size

                if frame_id == b'TIT2' and not tags['title']:
                    tags['title'] = _decode_text_field(data)
                elif frame_id == b'TPE1' and not tags['artist']:
                    tags['artist'] = _decode_text_field(data)
                elif frame_id == b'USLT' and not tags['lyrics_text']:
                    # encoding(1) + language(3) + 描述(以 \x00 结尾) + 歌词
                    if len(data) < 5:
                        continue
                    codec = _TEXT_CODECS.get(data[0], 'utf-8')
                    body = data[4:].decode(codec, errors='ignore')
                    tags['lyrics_text'] = body.split('\x00', 1)[1] if '\x00' in body else body

    except Exception as e:
        print(f'[LyricAligner] ID3 read error: {type(e).__name__}: {e}')

    if tags['title'] or tags['artist']:
        print(f"[LyricAligner] ID3 tags: {tags['title']!r} / {tags['artist']!r}")
    return tags


def _tags_from_filename(path):
    """
    ID3 没有标题时，从文件名猜测歌名与歌手
    形如「歌名-歌手.mp3」「歌手 - 歌名.mp3」，无法可靠区分，交给在线检索的相关性过滤兜底
    """
    stem = _AUDIO_EXT_RE.sub('', os.path.basename(path))
    stem = _DOWNLOAD_SUFFIX_RE.sub('', stem)
    stem = _BRACKET_RE.sub(' ', stem).strip()

    parts = [p.strip() for p in _SEPARATOR_RE.split(stem) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return stem, ''


def _find_lrc_file(vocal_path, original_path=None):
    for base_path in (original_path, vocal_path):
        if not base_path:
            continue
        candidate = os.path.splitext(base_path)[0] + '.lrc'
        if os.path.exists(candidate):
            print(f'[LyricAligner] Reading LRC: {candidate}')
            return candidate
    return None


def _read_lrc_file(lrc_path):
    try:
        with open(lrc_path, 'r', encoding='utf-8', errors='ignore') as f:
            return parse_lrc(f.read())
    except OSError as e:
        print(f'[LyricAligner] LRC read error: {e}')
        return {'lines': [], 'synced': False}


# ─── 人声时间轴 ───

def _audio_duration(vocal_path):
    try:
        import soundfile as sf
        info = sf.info(vocal_path)
        return float(info.duration)
    except Exception:
        _, duration = vocal_segments(vocal_path)
        return duration


def _distribute_over_vocals(lines, vocal_path):
    """
    把无时间戳的歌词铺到人声段落上
    先用最大余额法按段落时长把行分配到各段，保证一行不会跨过间奏，
    再在段内按字数权重切分时间
    """
    segments = singing_segments(vocal_path)
    total = sum(end - start for start, end in segments)
    if not segments or total <= 0:
        print('[LyricAligner] No vocal segments to distribute lyrics onto')
        return []

    quotas = [len(lines) * (end - start) / total for start, end in segments]
    counts = [int(q) for q in quotas]
    order = sorted(
        range(len(quotas)), key=lambda i: quotas[i] - counts[i], reverse=True
    )
    for i in range(len(lines) - sum(counts)):
        counts[order[i % len(order)]] += 1

    result = []
    cursor_line = 0
    for (seg_start, seg_end), count in zip(segments, counts):
        if count == 0:
            continue
        chunk = lines[cursor_line:cursor_line + count]
        cursor_line += count

        seg_len = seg_end - seg_start
        weights = [max(len(line['text'].strip()), 1) for line in chunk]
        weight_sum = sum(weights)

        current = seg_start
        for line, weight in zip(chunk, weights):
            end = min(current + seg_len * weight / weight_sum, seg_end)
            result.append({
                'text': line['text'],
                'start': round(current, 3),
                'end': round(max(end, current + 0.3), 3),
            })
            current = end

    print(
        f'[LyricAligner] Distributed {len(result)} lines over '
        f'{len(segments)} vocal segments'
    )
    return result

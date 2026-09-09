"""歌词链路自检：LRC 解析边界、ID3 USLT 各编码、本地 .lrc、同步歌词平移、文件名兜底"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.lrc_parser import parse_lrc, has_usable_lyrics
from engine.lyric_aligner import (
    _read_id3_tags, _tags_from_filename, _find_lrc_file,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f'      got : {got!r}')
        print(f'      want: {want!r}')
        failures.append(name)


# ─── 1. LRC 解析边界 ───
print('\n=== LRC 解析 ===')

p = parse_lrc('[ti:标题]\n[ar:歌手]\n[offset:+500]\n[00:12]整数秒\n[00:18.5]一位小数\n[00:24.123]三位毫秒\n')
check('元数据标签被过滤', [l['text'] for l in p['lines']], ['整数秒', '一位小数', '三位毫秒'])
check('整数秒时间戳', p['lines'][0]['start'], 11.5)   # offset +500ms => 提前 0.5s
check('一位小数时间戳', p['lines'][1]['start'], 18.0)
check('三位毫秒时间戳', p['lines'][2]['start'], 23.623)
check('标记为同步歌词', p['synced'], True)
check('中间行 end 接下一行 start', p['lines'][0]['end'], 18.0)

p = parse_lrc('[00:12.00][00:45.00]重复副歌\n[00:20.00]中间一句\n')
starts = sorted(l['start'] for l in p['lines'])
check('一行多时间戳展开', starts, [12.0, 20.0, 45.0])
check('多时间戳不残留方括号', all('[' not in l['text'] for l in p['lines']), True)

p = parse_lrc('[00:00.00] 作词 : 某人\n[00:01.00] 作曲 : 某人\n[00:05.00]纯音乐，请欣赏\n')
check('制作字幕与纯音乐被过滤', p['lines'], [])
check('无可用歌词判定', has_usable_lyrics(p), False)

p = parse_lrc('[00:00.00-1] 作曲 : Ling Lung\n')
check('网易云 -1 无时间轴标记', p['lines'], [])

p = parse_lrc('第一行没有时间戳\n第二行也没有\n')
check('非同步歌词保留文本', [l['text'] for l in p['lines']], ['第一行没有时间戳', '第二行也没有'])
check('非同步歌词无时间戳', p['lines'][0]['start'], None)
check('非同步标记', p['synced'], False)

check('空输入', parse_lrc('')['lines'], [])
check('空输入不算可用', has_usable_lyrics(parse_lrc('')), False)

p = parse_lrc(''.join(f'[00:{i:02d}.00]第{i}行\n' for i in range(1, 7)))
check('6 行算可用', has_usable_lyrics(p), True)
check('不足 4 行不算可用', has_usable_lyrics(parse_lrc('[00:01.00]a\n[00:02.00]b\n')), False)


# ─── 2. ID3 USLT 各编码 ───
print('\n=== ID3 USLT ===')

LYRICS = '[00:12.00]第一句歌词\n[00:18.00]第二句歌词\n[00:24.00]第三句歌词\n[00:30.00]第四句歌词'


def build_id3(frames, version=3):
    body = b''
    for frame_id, data in frames:
        n = len(data)
        if version == 4:
            size = bytes([(n >> 21) & 0x7f, (n >> 14) & 0x7f, (n >> 7) & 0x7f, n & 0x7f])
        else:
            size = struct.pack('>I', n)
        body += frame_id + size + b'\x00\x00' + data
    n = len(body)
    tag_size = bytes([(n >> 21) & 0x7f, (n >> 14) & 0x7f, (n >> 7) & 0x7f, n & 0x7f])
    return b'ID3' + bytes([version, 0, 0]) + tag_size + body


def write_mp3(tag, path):
    with open(path, 'wb') as f:
        f.write(tag)
        f.write(b'\xff\xfb\x90\x00' + b'\x00' * 500)  # 一段假的 MPEG 帧同步头


def uslt(encoding, description, text):
    if encoding == 0:
        payload = description.encode('latin-1') + b'\x00' + text.encode('latin-1')
    elif encoding == 1:
        payload = (description + '\x00' + text).encode('utf-16')          # 带 BOM
    elif encoding == 2:
        payload = (description + '\x00' + text).encode('utf-16-be')       # 无 BOM
    else:
        payload = description.encode('utf-8') + b'\x00' + text.encode('utf-8')
    return bytes([encoding]) + b'chi' + payload


tmpdir = tempfile.mkdtemp()
cases = [
    ('v2.3 UTF-8 空描述', 3, 3, '', LYRICS),
    ('v2.3 UTF-8 有描述', 3, 3, 'Lyrics', LYRICS),
    ('v2.3 UTF-16 BOM 空描述', 3, 1, '', LYRICS),
    ('v2.3 UTF-16 BOM 有描述', 3, 1, '歌词', LYRICS),
    ('v2.3 UTF-16BE 无 BOM', 3, 2, '', LYRICS),
    ('v2.4 synchsafe 帧长', 4, 3, '', LYRICS),
]

for name, version, encoding, description, text in cases:
    path = os.path.join(tmpdir, f'case_{version}_{encoding}_{len(description)}.mp3')
    tag = build_id3([
        (b'TIT2', bytes([3]) + '测试歌曲'.encode('utf-8')),
        (b'TPE1', bytes([1]) + '测试歌手'.encode('utf-16')),
        (b'USLT', uslt(encoding, description, text)),
    ], version=version)
    write_mp3(tag, path)
    tags = _read_id3_tags(path)
    check(f'{name} - 标题', tags['title'], '测试歌曲')
    check(f'{name} - 歌手', tags['artist'], '测试歌手')
    parsed = parse_lrc(tags['lyrics_text'])
    check(f'{name} - 歌词行数', len(parsed['lines']), 4)
    check(f'{name} - 首行文本', parsed['lines'][0]['text'] if parsed['lines'] else None, '第一句歌词')

# 无 USLT 时不应报错，且要能拿到标题歌手
path = os.path.join(tmpdir, 'no_uslt.mp3')
write_mp3(build_id3([(b'TIT2', bytes([1]) + '只有标题'.encode('utf-16'))]), path)
tags = _read_id3_tags(path)
check('无 USLT 帧 - 标题', tags['title'], '只有标题')
check('无 USLT 帧 - 歌词为空', tags['lyrics_text'], '')

check('文件不存在', _read_id3_tags('/tmp/__definitely_missing__.mp3')['title'], '')
check('original_path 为 None', _read_id3_tags(None)['title'], '')

# 截断/损坏的标签不应抛异常
path = os.path.join(tmpdir, 'corrupt.mp3')
with open(path, 'wb') as f:
    f.write(b'ID3\x03\x00\x00' + bytes([0, 0, 0x7f, 0x7f]) + b'\x01\x02')
check('损坏标签不抛异常', _read_id3_tags(path), {'title': '', 'artist': '', 'lyrics_text': ''})


# ─── 3. 文件名兜底 ───
print('\n=== 文件名兜底 ===')

check('歌名-歌手', _tags_from_filename('/x/山风山风等等我-万海东.mp3'), ('山风山风等等我', '万海东'))
check('下载站随机后缀', _tags_from_filename('/x/山风山风等等我-万海东#2RCUbL.mp3'), ('山风山风等等我', '万海东'))
check('全角破折号不误切', _tags_from_filename('/x/太阳——熟透的苹果.mp3'), ('太阳——熟透的苹果', ''))
check('分离产物名还原', _tags_from_filename('/x/山风山风等等我-万海东_ef3cb4f8e0d5.vocals.wav'), ('山风山风等等我', '万海东'))
check('括号版本标记剥离', _tags_from_filename('/x/某歌 (DJ版)-某人.mp3'), ('某歌', '某人'))
check('无分隔符', _tags_from_filename('/x/纯歌名.mp3'), ('纯歌名', ''))


# ─── 4. 本地 .lrc 查找 ───
print('\n=== 本地 .lrc ===')

lrc_dir = tempfile.mkdtemp()
lrc_path = os.path.join(lrc_dir, '某首歌.lrc')
with open(lrc_path, 'w', encoding='utf-8') as f:
    f.write('[00:10.00]本地歌词第一行\n[00:15.00]本地歌词第二行\n')
check('按原始 mp3 同目录找到', _find_lrc_file('/nonexistent/vocal.wav', os.path.join(lrc_dir, '某首歌.mp3')), lrc_path)

wav_dir = tempfile.mkdtemp()
wav_path = os.path.join(wav_dir, 'vocal.wav')
open(wav_path, 'wb').close()
with open(os.path.join(wav_dir, 'vocal.lrc'), 'w', encoding='utf-8') as f:
    f.write('[00:10.00]x\n')
check('按人声 wav 同目录找到', _find_lrc_file(wav_path, None), os.path.join(wav_dir, 'vocal.lrc'))
check('原始 mp3 优先于人声 wav', _find_lrc_file(wav_path, os.path.join(lrc_dir, '某首歌.mp3')), lrc_path)
check('找不到时返回 None', _find_lrc_file('/nonexistent/a.wav', '/nonexistent/a.mp3'), None)

real_lrc = os.path.expanduser('~/Music/网易云音乐/张艾嘉 - 童年.lrc')
if os.path.exists(real_lrc):
    with open(real_lrc, 'r', encoding='utf-8', errors='ignore') as f:
        parsed = parse_lrc(f.read())
    usable = has_usable_lyrics(parsed)
    print(f'PASS  真实 LRC 文件可解析: {len(parsed["lines"])} 行, synced={parsed["synced"]}')
    if not usable:
        failures.append('真实 LRC 文件解析')
    if parsed['lines']:
        print(f'      首行: {parsed["lines"][0]}')
else:
    print('SKIP  真实 LRC 文件不存在')


# ─── 5. 人声段落检测 ───
print('\n=== 人声段落检测 ===')

from engine.vocal_segments import vocal_segments

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
cases = [
    ('黄昏', 343.9),
    ('山风山风等等我', 209.4),
    ('太阳', 144.8),
]
for prefix, expected_duration in cases:
    matches = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.endswith('.vocals.wav')
    ) if os.path.isdir(output_dir) else []
    if not matches:
        print(f'SKIP  {prefix}: 没有分离产物')
        continue
    segs, dur = vocal_segments(matches[0])
    ok = len(segs) > 0 and abs(dur - expected_duration) < 1.0
    print(f"{'PASS' if ok else 'FAIL'}  {prefix}: 段数={len(segs)} 时长={dur:.1f}s "
          f"有声={sum(e - s for s, e in segs):.1f}s")
    if not ok:
        failures.append(f'人声段落检测 {prefix}')
    # 段落必须有序且不重叠
    ordered = all(segs[i][1] <= segs[i + 1][0] for i in range(len(segs) - 1))
    check(f'{prefix} 段落有序不重叠', ordered, True)


print('\n' + '=' * 50)
if failures:
    print(f'{len(failures)} 项失败:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
print('全部通过')

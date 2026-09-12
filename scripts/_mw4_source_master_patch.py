from pathlib import Path

WORKFLOWS = [
    Path('.github/workflows/mw4-semantic-v3-1-shadow.yml'),
    Path('.github/workflows/mw4-fullframe-retention-v3-1.yml'),
]

RESOLVER = '''              derivative_items = list(asset.get('derivatives') or [])
              source_candidates = []
              for item in derivative_items:
                  if str(item.get('type') or '').lower() != 'source':
                      continue
                  candidate_url = item.get('url') or (item.get('properties') or {}).get('url')
                  if not candidate_url:
                      continue
                  try:
                      file_size = int(item.get('fileSize') or 0)
                  except (TypeError, ValueError):
                      file_size = 0
                  source_candidates.append((file_size, item, candidate_url))

              if not source_candidates:
                  available_types = sorted({str(item.get('type')) for item in derivative_items})
                  raise RuntimeError(
                      f'Original MediaSilo source master unavailable for {asset["title"]}; '
                      f'refusing x1080/proxy fallback. available derivative types={available_types}'
                  )

              _, derivative, url = max(source_candidates, key=lambda entry: entry[0])
              if str(derivative.get('type') or '').lower() != 'source':
                  raise RuntimeError('internal error: selected MediaSilo derivative is not type=source')

              Path('/tmp/resolved_source.json').write_text(
                  json.dumps({
                      'title': asset['title'],
                      'file_name': asset.get('fileName'),
                      'url': url,
                      'derivative_type': 'source',
                      'file_size': derivative.get('fileSize'),
                      'width': derivative.get('width'),
                      'height': derivative.get('height'),
                      'duration_ms': derivative.get('duration'),
                  }),
                  encoding='utf-8',
              )
'''

SOURCE_QA = '''          if selected.get('derivative_type') != 'source':
              raise RuntimeError('refusing non-source MediaSilo derivative before semantic analysis')

          actual_size = target.stat().st_size
          expected_size = int(selected.get('file_size') or 0)
          if expected_size and actual_size != expected_size:
              raise RuntimeError(
                  f'original master byte-size mismatch: downloaded={actual_size} expected={expected_size}'
              )

          probe = json.loads(subprocess.check_output([
              'ffprobe', '-v', 'error',
              '-show_entries',
              'stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,bit_rate:'
              'format=duration,size,bit_rate,format_name',
              '-of', 'json', str(target),
          ], text=True))
          video = next(item for item in probe['streams'] if item.get('codec_type') == 'video')
          if (video.get('width'), video.get('height')) != (1920, 1080):
              raise RuntimeError(f'original master geometry mismatch: {video.get("width")}x{video.get("height")}')
          if video.get('avg_frame_rate') != '60000/1001':
              raise RuntimeError(f'original master FPS mismatch: {video.get("avg_frame_rate")}')

          declared_width = selected.get('width')
          declared_height = selected.get('height')
          if declared_width and int(declared_width) != 1920:
              raise RuntimeError(f'MediaSilo source metadata width mismatch: {declared_width}')
          if declared_height and int(declared_height) != 1080:
              raise RuntimeError(f'MediaSilo source metadata height mismatch: {declared_height}')

          import hashlib
          digest = hashlib.sha256()
          with target.open('rb') as stream:
              while block := stream.read(8 * 1024 * 1024):
                  digest.update(block)

          source_qa = {
              'source_key': source_key,
              'title': selected.get('title'),
              'file_name': selected.get('file_name'),
              'derivative_type': 'source',
              'proxy_fallback_allowed': False,
              'downloaded_bytes': actual_size,
              'declared_source_bytes': expected_size or None,
              'sha256': digest.hexdigest(),
              'probe': probe,
          }
          qa_path = Path('raw_sources') / f'{source_key}.source_qa.json'
          qa_path.write_text(json.dumps(source_qa, indent=2), encoding='utf-8')
          print(
              f'ORIGINAL SOURCE MASTER VERIFIED type=source bytes={actual_size} '
              f'fps={video.get("avg_frame_rate")} codec={video.get("codec_name")} '
              f'sha256={source_qa["sha256"][:16]}...'
          )
'''

STAGED_QA = '''          source = Path('raw_sources') / f'{source_key}.mp4'
          source_qa_path = Path('raw_sources') / f'{source_key}.source_qa.json'
          if not source.is_file():
              raise RuntimeError(f'staged source artifact missing: {source}')
          if not source_qa_path.is_file():
              raise RuntimeError(f'original-source QA manifest missing: {source_qa_path}')

          source_qa = json.loads(source_qa_path.read_text(encoding='utf-8'))
          if source_qa.get('derivative_type') != 'source':
              raise RuntimeError('staged reel was not certified as MediaSilo type=source')
          if source_qa.get('proxy_fallback_allowed') is not False:
              raise RuntimeError('staged reel source policy allows proxy fallback')
          if source.stat().st_size != int(source_qa.get('downloaded_bytes') or 0):
              raise RuntimeError('staged source byte-size differs from original-source QA manifest')

          import hashlib
          digest = hashlib.sha256()
          with source.open('rb') as stream:
              while block := stream.read(8 * 1024 * 1024):
                  digest.update(block)
          if digest.hexdigest() != source_qa.get('sha256'):
              raise RuntimeError('staged source SHA256 differs from original-source QA manifest')

          probe = json.loads(subprocess.check_output([
              'ffprobe', '-v', 'error',
              '-show_entries', 'stream=width,height,avg_frame_rate:format=duration',
              '-of', 'json', str(source),
          ], text=True))
          video = next(item for item in probe['streams'] if item.get('width'))
          assert (video['width'], video['height']) == (1920, 1080)
          assert video['avg_frame_rate'] == '60000/1001'
          print(
              f'staged ORIGINAL source verified bytes={source.stat().st_size} '
              f'sha256={source_qa["sha256"][:16]}...'
          )
'''


def patch(path: Path) -> None:
    text = path.read_text(encoding='utf-8')

    resolver_start_token = "              derivatives = {item.get('type'): item for item in asset.get('derivatives', [])}"
    resolver_start = text.index(resolver_start_token)
    resolver_end = text.index("              break\n          else:", resolver_start)
    text = text[:resolver_start] + RESOLVER + text[resolver_end:]

    download_anchor = text.index("          selected = json.loads(Path('/tmp/resolved_source.json').read_text(encoding='utf-8'))")
    qa_start = text.index("          probe = json.loads(subprocess.check_output([", download_anchor)
    qa_end = text.index("          PY", qa_start)
    text = text[:qa_start] + SOURCE_QA + text[qa_end:]

    old_upload = '          path: raw_sources/${{ matrix.source }}.mp4'
    new_upload = (
        '          path: |\n'
        '            raw_sources/${{ matrix.source }}.mp4\n'
        '            raw_sources/${{ matrix.source }}.source_qa.json'
    )
    if text.count(old_upload) != 1:
        raise RuntimeError(f'{path}: expected exactly one source artifact upload path, got {text.count(old_upload)}')
    text = text.replace(old_upload, new_upload, 1)

    verify_step = text.index('      - name: Verify staged source reel before')
    verify_start = text.index("          source = Path('raw_sources') / f'{source_key}.mp4'", verify_step)
    verify_end = text.index('          PY', verify_start)
    text = text[:verify_start] + STAGED_QA + text[verify_end:]

    if "derivatives.get('x1080')" in text or "derivatives.get('proxy')" in text:
        raise RuntimeError(f'{path}: x1080/proxy fallback remains')
    if "'derivative_type': 'source'" not in text:
        raise RuntimeError(f'{path}: source certification missing')
    if '.source_qa.json' not in text:
        raise RuntimeError(f'{path}: source QA manifest is not staged')
    if 'video_bitrate_kbps' in text and '250000' not in text:
        raise RuntimeError(f'{path}: unexpected output-quality mutation')

    path.write_text(text, encoding='utf-8')
    print(f'patched {path}')


for workflow in WORKFLOWS:
    patch(workflow)

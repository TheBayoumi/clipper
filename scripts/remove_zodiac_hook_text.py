from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

WIDTH=1080
HEIGHT=1920
FPS=60
BITRATE='250M'


def run(cmd:list[str])->None:
    print('+',' '.join(cmd))
    subprocess.run(cmd,check=True)


def x264()->list[str]:
    return [
        '-c:v','libx264','-preset','slow','-profile:v','high','-level:v','5.2',
        '-b:v',BITRATE,'-minrate',BITRATE,'-maxrate',BITRATE,'-bufsize','500M',
        '-x264-params','nal-hrd=cbr:force-cfr=1','-r',str(FPS),'-fps_mode','cfr',
        '-g','120','-keyint_min','60','-sc_threshold','0','-pix_fmt','yuv420p',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv'
    ]


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--work-dir',type=Path,default=Path('zodiac-map-render-v2'))
    p.add_argument('--master',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()

    work=a.work_dir.resolve()
    assets=work/'assets'
    temp=work/'textless'
    temp.mkdir(parents=True,exist_ok=True)
    art=assets/'zodiac_wz_1080x1920.png'
    map_file=assets/'map_flythrough_9x16.mp4'
    master=a.master.resolve()
    output=a.output.resolve()

    s0=temp/'S0_clean.mp4'
    s1=temp/'S1_clean.mp4'
    tail=temp/'tail_clean.mp4'
    video_only=temp/'video_only.mp4'
    audio=temp/'audio.m4a'
    manifest=temp/'concat.txt'

    # 0.00-0.60: native 9:16 art, same subtle motion, NO added hook text.
    vf0=(
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(zoom+0.0012,1.035)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p"
    )
    run(['ffmpeg','-y','-loop','1','-i',str(art),'-t','0.600','-vf',vf0,*x264(),'-an',str(s0)])

    # 0.60-1.60: original vertical map footage, NO added campaign hook text.
    vf1=(
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps={FPS},format=yuv420p"
    )
    run(['ffmpeg','-y','-ss','0.600','-i',str(map_file),'-t','1.000','-vf',vf1,*x264(),'-an',str(s1)])

    # 1.60-end: preserve the already approved visual edit exactly, only re-encode for concat compatibility.
    run(['ffmpeg','-y','-ss','1.600','-i',str(master),'-t','9.000','-map','0:v:0','-vf',f'fps={FPS},format=yuv420p',*x264(),'-an',str(tail)])

    manifest.write_text(''.join(f"file '{x.resolve()}'\n" for x in (s0,s1,tail)))
    run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(manifest),'-map','0:v:0','-c:v','copy','-an',str(video_only)])

    # Preserve the already-fixed continuous source audio unchanged.
    run(['ffmpeg','-y','-i',str(master),'-map','0:a:0','-c:a','copy',str(audio)])
    run(['ffmpeg','-y','-i',str(video_only),'-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','copy','-shortest','-movflags','+faststart',str(output)])
    return 0


if __name__=='__main__':
    raise SystemExit(main())

from scripts.tjr_media_policy import (
    has_production_hd_video_stream,
    production_hd_dimensions,
    production_hd_format,
)


def test_production_hd_requires_both_720_short_side_and_1280_long_side() -> None:
    assert production_hd_dimensions(1280, 720)
    assert production_hd_dimensions(720, 1280)
    assert production_hd_dimensions(1920, 1080)
    assert not production_hd_dimensions(960, 720)
    assert not production_hd_dimensions(720, 960)
    assert not production_hd_dimensions(1280, 640)


def test_ytdlp_and_ffprobe_use_the_same_hd_rule() -> None:
    assert production_hd_format({"width": 1280, "height": 720, "vcodec": "avc1"})
    assert not production_hd_format({"width": 960, "height": 720, "vcodec": "avc1"})
    assert not production_hd_format({"width": 1920, "height": 1080, "vcodec": "none"})
    assert has_production_hd_video_stream([{"codec_type": "video", "width": 1920, "height": 1080}])
    assert not has_production_hd_video_stream(
        [{"codec_type": "video", "width": 960, "height": 720}]
    )

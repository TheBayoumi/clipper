from pathlib import Path


def test_v10_public_youtube_acquisition_activates_bgutil_before_optional_cookies() -> None:
    source = Path("scripts/modal_v10_cycle.py").read_text(encoding="utf-8")
    assert "youtubepot-bgutilscript:" in source
    assert "server_home=/root/bgutil-ytdlp-pot-provider/server" in source
    assert "youtube:player_client=default,mweb" in source
    assert "youtube:player_client=web_embedded,android_vr" in source
    assert '"yt-dlp[default]>=2026.7.4,<2027"' in source
    assert '"--js-runtimes"' in source
    assert '"node"' in source
    assert '"bgutil_default_mweb"' in source
    assert '"cookies_bgutil_default_mweb"' in source
    assert source.index('"bgutil_default_mweb"') < source.index('"cookies_bgutil_default_mweb"')
    assert "Authenticated yt-dlp cookies are required from this cloud egress." not in source


def test_open_model_image_has_qwen3_vl_torchvision_runtime_contract() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert '"torch==2.8.0"' in source
    assert '"torchvision==0.23.0"' in source
    assert 'if task.startswith("story_moments:"):' in source
    assert "return 1024" in source

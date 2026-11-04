"""PlaywrightGamePage + clock.js against a tiny local WebGL page (never the real game)."""

from dataclasses import replace

import pytest

from qwop_agent.config import GameConfig, Region
from qwop_agent.page import GamePageError, PlaywrightGamePage

pytestmark = pytest.mark.browser

# A stand-in "game": a light grey start screen until clicked, then the canvas
# alternates red (odd frames) and blue (even frames). It records key events
# and the time between animation frames.
TEST_PAGE = """<!doctype html>
<html><body style="margin:0">
<canvas width="640" height="400"></canvas>
<script>
const canvas = document.querySelector("canvas");
const gl = canvas.getContext("webgl");
Object.assign(window, {framesRun: 0, started: false, keyLog: [], gaps: []});
document.addEventListener("keydown", (e) => keyLog.push("+" + e.key));
document.addEventListener("keyup", (e) => keyLog.push("-" + e.key));
canvas.addEventListener("click", () => { started = true; });
let last = null;
function frame(t) {
  if (last !== null) gaps.push(t - last);
  last = t;
  framesRun++;
  if (!started) gl.clearColor(237 / 255, 237 / 255, 237 / 255, 1);
  else if (framesRun % 2) gl.clearColor(0.8, 0.1, 0.1, 1);
  else gl.clearColor(0.1, 0.1, 0.8, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body></html>
"""

WHOLE = Region(0.0, 0.0, 1.0, 1.0, size=(16, 10))


def chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as playwright:
            playwright.chromium.launch().close()
    except Exception:  # noqa: BLE001 - browser not installed
        return False
    return True


if not chromium_available():
    pytest.skip("Playwright's Chromium is not installed", allow_module_level=True)


@pytest.fixture
def page(tmp_path):
    html = tmp_path / "game.html"
    html.write_text(TEST_PAGE, encoding="utf-8")
    game = PlaywrightGamePage(GameConfig(url=html.as_uri(), viewport=(700, 450), load_timeout_s=20))
    game.open()
    yield game
    game.close()


def js(game, expression):
    return game._page.evaluate(expression)


def test_open_waits_for_the_start_screen_and_clicks_it(page):
    assert page.is_open
    assert page.loads == 1
    assert js(page, "started") is True


def test_frames_only_advance_when_asked_with_a_fixed_step(page):
    before = js(page, "framesRun")
    page.advance(5, {})
    assert js(page, "framesRun") == before + 5
    gaps = js(page, "gaps.slice(-5)")
    assert gaps == pytest.approx([1000 / 60] * 5)
    # Real time passing does not move the game.
    page._page.wait_for_timeout(200)
    assert js(page, "framesRun") == before + 5


def test_capture_shows_the_last_frame(page):
    for frames in (3, 4):
        shot = page.advance(frames, {"whole": WHOLE})["whole"]
        assert shot.shape == (10, 16, 3)
        red, _, blue = shot.reshape(-1, 3).mean(axis=0)
        last_frame_was_odd = js(page, "framesRun") % 2 == 1
        assert (red > blue) == last_frame_was_odd


def test_hold_presses_and_releases_only_what_changed(page):
    page.hold(frozenset("qo"))
    page.hold(frozenset("qp"))
    page.hold(frozenset())
    assert js(page, "keyLog") == ["+o", "+q", "-o", "+p", "-p", "-q"]


def test_restart_releases_keys_and_presses_r(page):
    page.hold(frozenset("w"))
    page.restart()
    assert js(page, "keyLog") == ["+w", "-w", "+r", "-r"]


def test_closed_page_raises(page):
    page.close()
    assert not page.is_open
    with pytest.raises(GamePageError):
        page.advance(1, {})


def test_unreachable_game_raises_a_clear_error(tmp_path):
    missing = PlaywrightGamePage(GameConfig(url=(tmp_path / "missing.html").as_uri(), load_timeout_s=5))
    with pytest.raises(GamePageError):
        missing.open()
    assert not missing.is_open


def test_realtime_mode_paces_frames(page):
    paced = replace(page.config, realtime=True)
    page.config = paced
    page.advance(1, {})
    started = js(page, "Date.now()")
    page.advance(12, {})  # 12 frames at 60 fps = 200 ms
    page.advance(12, {})
    assert js(page, "Date.now()") - started >= 190

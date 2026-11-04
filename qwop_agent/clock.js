// Virtual clock, injected before the game page loads.
//
// The game draws a frame whenever the browser calls requestAnimationFrame and
// measures time with performance.now(). Both are standard browser APIs; we
// replace them with a clock that only moves when Python asks for frames. That
// makes every environment step cover exactly the same amount of game time, no
// matter how busy the machine is, and lets a fast machine play faster than
// real time. The game's own code is not touched.
//
// Two more tricks, both on standard APIs:
// * Frames that nobody will look at (all but the last one of each advance)
//   skip their WebGL draw calls. The game's physics still runs every frame,
//   so the race is identical; we just don't paint pictures nobody sees.
// * Captures happen in the same task as the last frame, because a WebGL
//   canvas clears its drawing buffer once the browser has shown it.
(() => {
  let now = 0;
  let queue = [];
  let nextId = 1;
  let drawing = true;

  window.requestAnimationFrame = (callback) => {
    const id = nextId++;
    queue.push([id, callback]);
    return id;
  };
  window.cancelAnimationFrame = (id) => {
    queue = queue.filter((entry) => entry[0] !== id);
  };
  performance.now = () => now;

  const contexts = [window.WebGLRenderingContext, window.WebGL2RenderingContext].filter(Boolean);
  for (const { prototype } of contexts) {
    for (const name of ["clear", "drawArrays", "drawElements"]) {
      const original = prototype[name];
      prototype[name] = function (...args) {
        if (drawing) return original.apply(this, args);
      };
    }
  }

  const snapshot = document.createElement("canvas");
  const snapshotContext = snapshot.getContext("2d", { willReadFrequently: true });
  const scratch = document.createElement("canvas");
  const scratchContext = scratch.getContext("2d", { willReadFrequently: true });

  function toBase64(bytes) {
    const chunk = 0x8000;
    let binary = "";
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }

  function crop(region) {
    const [x, y, w, h, outW, outH] = region;
    scratch.width = outW;
    scratch.height = outH;
    scratchContext.drawImage(snapshot, x, y, w, h, 0, 0, outW, outH);
    return toBase64(scratchContext.getImageData(0, 0, outW, outH).data);
  }

  window.__qwopClock = {
    advance(frames, frameMs, regions) {
      for (let i = 0; i < frames; i++) {
        now += frameMs;
        drawing = i === frames - 1;
        const due = queue;
        queue = [];
        try {
          for (const [, callback] of due) callback(now);
        } finally {
          drawing = true;
        }
      }
      const source = document.querySelector("canvas");
      const shots = {};
      const names = Object.keys(regions || {});
      if (source && names.length) {
        // Copy the WebGL canvas once, then cut every region from the copy.
        snapshot.width = source.width;
        snapshot.height = source.height;
        snapshotContext.drawImage(source, 0, 0);
        for (const name of names) shots[name] = crop(regions[name]);
      }
      return {
        now,
        canvas: source ? [source.width, source.height] : null,
        shots,
      };
    },
  };
})();

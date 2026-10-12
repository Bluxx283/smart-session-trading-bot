# Screen Capture Integration — Readiness and Safe Next Steps

## Current status

SSAD currently analyzes market-provider OHLCV candles and overlays statistical anomaly markers. It does **not** capture the desktop, a broker window, or chart pixels. The dashboard now labels this explicitly.

## Recommended implementation sequence

1. **Choose a capture source**: a user-selected window or monitor. For a local Python app, use an OS capture library such as `mss`; for browser-mediated capture, use `getDisplayMedia()` only after an explicit click and user permission.
2. **Keep capture opt-in**: show the active source, capture status, last-frame time, and a visible Stop control. Never start capture silently.
3. **Add a frame pipeline**: timestamp each frame in UTC, resize/downsample it, and discard frames when capture is stopped. Avoid storing screenshots by default.
4. **Validate before inference**: check for a blank/occluded frame, expected chart region, readable symbol/timeframe, and OCR confidence. Do not convert uncertain pixels into trade signals.
5. **Compare against known states**: test chart symbol, displayed price, and candle direction against the existing provider feed; report mismatches instead of silently choosing one source.
6. **Keep actions safe**: screenshot-derived findings are informational until measured on a labelled test set. Leave automated orders disabled.

## Acceptance checklist

- [ ] User can start capture only after selecting a source and granting permission.
- [ ] UI clearly displays Capturing / Stopped / Permission denied / No frames.
- [ ] Last-frame timestamp is shown in UTC and stale frames trigger a warning.
- [ ] Stop immediately ends frame processing.
- [ ] No screenshots or secrets are persisted by default.
- [ ] Tests cover permission failure, missing frames, stale frames, and source mismatch.
- [ ] No order can be triggered by a screenshot-only result.

## Why this is the next step

The quote and candlestick chart already work, and the anomaly feed returns some detections. The largest unknown is whether SSAD can observe the same chart the user sees. Resolve that with a visible, permission-based capture prototype before expanding anomaly logic or connecting execution.

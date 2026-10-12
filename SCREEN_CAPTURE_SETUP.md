# Screen Capture Integration — Readiness and Safe Next Steps

## Current status

SSAD analyzes market-provider OHLCV candles and overlays statistical anomaly markers. The dashboard now includes an **opt-in, one-shot local display capture prototype**: after checking a consent box, the user chooses a display and clicks Capture. The frame is encoded in memory, previewed in the Streamlit session, and timestamped in UTC; the helper does not save it to disk or run in the background. Install `mss` via `requirements.txt`.

**Important:** run Streamlit locally on the same computer as TradingView. A remotely hosted Streamlit app captures the server display, not the user's desktop. This prototype does not yet read chart text or detect patterns.

## Recommended next implementation sequence

1. **Verify one-shot capture locally**: open TradingView, run `streamlit run app.py` on that same computer, consent, and capture the display showing the chart.
2. **Add OCR with confidence reporting**: extract symbol, timeframe, and visible price; do not accept low-confidence readings as facts.
3. **Add visual checks**: chart-region presence, blank/occluded frame detection, and basic candle/shape analysis.
4. **Compare against known states**: test chart symbol, displayed price, and candle direction against the existing provider feed; report mismatches instead of silently choosing one source.
5. **Keep actions safe**: screenshot-derived findings remain informational until measured on a labelled test set. Leave automated orders disabled.

## Acceptance checklist

- [x] User must explicitly consent and click Capture for a one-shot frame.
- [x] UI shows a preview, UTC timestamp, and capture errors.
- [x] The capture helper encodes in memory and does not save screenshots to disk.
- [ ] UI supports an active Capture / Stop lifecycle for continuous monitoring.
- [ ] OCR extracts symbol, timeframe, and price with confidence scores.
- [ ] Stale frames and occluded charts are detected.
- [ ] Tests cover permission failure, missing frames, stale frames, and source mismatch.
- [ ] No order can be triggered by a screenshot-only result.

## Why this is the next step

The quote and candlestick chart already work, and the anomaly feed returns some detections. The largest unknown is whether SSAD can observe the same chart the user sees. Resolve that with a visible, permission-based capture prototype before expanding anomaly logic or connecting execution.

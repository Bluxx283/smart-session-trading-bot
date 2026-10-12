# SSAD Floating AI Companion (prototype)

This is an optional desktop companion for the existing SSAD app. It is a small,
draggable, translucent, always-on-top window so you can keep your broker/chart visible
while asking for a cautious explanation of the visible chart.

## Install and run

Use a local desktop computer (not a remote Streamlit host). From the repository folder:

```bash
python -m pip install -r requirements.txt
```

Set an API key in your environment, then start the companion:

**Windows PowerShell**
```powershell
$env:OPENAI_API_KEY="your-api-key"
python desktop_copilot.py
```

**macOS/Linux**
```bash
export OPENAI_API_KEY="your-api-key"
python desktop_copilot.py
```

Optional: set `SSAD_VISION_MODEL` to a vision-capable model available to your API account.
The default is `gpt-4o-mini`.

## Privacy and controls

- The companion captures nothing until you check the consent box and click **Analyze screen**.
- It captures one selected physical display per click. It does not run continuous/background capture.
- The screenshot is encoded in memory, sent to the configured OpenAI API for analysis, and not saved to disk by this code. The provider processes the image under your API account's terms.
- Avoid displaying passwords, account identifiers, private messages, API keys, or other sensitive information on the selected display.
- You can drag the window, toggle opacity, clear the chat, or close it at any time.
- This prototype does not place or manage broker orders, and does not provide guaranteed signals. Chart interpretation can be wrong; verify any scenario and risk levels independently.

## Current prototype limits

This is a desktop companion beside SSAD, not yet an embedded control inside the Streamlit dashboard. It currently supports one-shot screenshot analysis and typed questions. Window-specific capture, microphone input, speech output, signal synchronization back into the SSAD dashboard, robust market-data cross-checks, and automated trade execution are not implemented. Trade execution should remain disabled while the analysis workflow is validated.

Screen capture permissions and multi-monitor behavior vary by operating system. On macOS, grant screen recording permission to the terminal/Python app. On Windows, capture is limited to displays accessible to the current desktop session.

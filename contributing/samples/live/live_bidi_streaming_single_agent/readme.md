# Simplistic Live (Bidi-Streaming) Agent

This project provides a basic example of a live, [bidirectional streaming](https://google.github.io/adk-docs/streaming/) agent
designed for testing and experimentation.

## Getting Started

Follow these steps to get the agent up and running:

1. **Start the ADK Web Server**
   Open your terminal, navigate to the root directory that contains the
   `live_bidi_streaming_single_agent` folder, and execute the following command:

   ```bash
   adk web
   ```

1. **Access the ADK Web UI**
   Once the server is running, open your web browser and navigate to the URL
   provided in the terminal (it will typically be `http://localhost:8000`).

1. **Select the Agent**
   In the top-left corner of the ADK Web UI, use the dropdown menu to select
   this agent.

1. **Start Streaming**
   Click on either the **Audio** or **Video** icon located near the chat input
   box to begin the streaming session.

1. **Interact with the Agent**
   You can now begin talking to the agent, and it will respond in real-time.

## Evaluating over audio

Eval cases describe *what* is tested (a conversation scenario or a fixed
script) and stay transport-agnostic — the same case can run as text or voice.
*How* a run is voiced (voice, language, transport, background noise, speaking
rate, barge-in) is run configuration, supplied separately via a live run-config
file and applied uniformly across the run's cases.

To run `eval_set_1` over native audio with the voice settings in
`live_run_config.json`:

```bash
adk eval \
  path/to/live_bidi_streaming_single_agent \
  path/to/live_bidi_streaming_single_agent/eval_set_1.evalset.json \
  --live_run_config_file path/to/live_bidi_streaming_single_agent/live_run_config.json
```

The `transport` in the run-config file selects the audio transport, so an
explicit `--live_transport` flag is not required. Audio runs use reference-free
live metrics (local latency by default; add `--managed_metrics` for the managed
Gen AI Eval Service multi-turn metrics, which require a GCP project).

## Usage Notes

- You only need to click the **Audio** or **Video** button once to initiate the
  stream. The current version does not support stopping and restarting the stream
  by clicking the button again during a session.

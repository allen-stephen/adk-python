# Model Armor plugin — text (unary) sample

Demonstrates screening a text agent's input and output with Google Cloud
Model Armor via the `ModelArmorPlugin`.

## What it shows

- Enable Model Armor screening once, at the app level, with a single config.
- The plugin screens **user input** (before the model) and **model output**
  (before it reaches the user) using Model Armor's synchronous API.
- `EnforcementMode.BLOCK` substitutes a safe reply on a policy match;
  `EnforcementMode.OBSERVE` passes content through but logs the verdict.

## Prerequisites

```bash
pip install 'google-adk[gcp]'
gcloud auth application-default login
```

Create Model Armor prompt and response templates in a region, then edit
`agent.py` to point `project`, `location`, `prompt_template_name`, and
`response_template_name` at your resources.

## Run

```bash
adk run contributing/samples/plugin/model_armor_unary
```

## Zero-code alternative

If you only need **server-side text** screening, the model platform can screen
prompts/responses natively via `generate_content_config.model_armor_config` —
no plugin required. Use this plugin when you also need **voice (live)**
coverage, **custom blocked responses**, or **log-only (OBSERVE)** monitoring,
which the platform-native path does not offer.

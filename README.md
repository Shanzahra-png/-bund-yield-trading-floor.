# Bund Yield Trading Floor

A simulated trading environment where an AI reads live market conditions and gives a real-time trading signal — powered by **FinBERT-GF**, a hybrid deep learning model built for my MS thesis on ECB monetary policy tone classification. No real money is involved; this is a paper-trading demo.

![Demo](https://youtu.be/DUH1h3EaXYc)

## What this is

The app simulates a live-moving **DE10Y (German 10-year Bund yield)** tape alongside a feed of ECB-style policy headlines. On each headline, the trained FinBERT-GF model classifies its tone — **Dovish / Neutral / Hawkish** — and surfaces that as a trading signal. The user trades manually against that signal using Buy/Sell controls and a virtual cash balance; nothing here touches real funds or a real exchange.

The instrument and headlines were deliberately kept in-domain: FinBERT-GF's temporal branch is trained on real Bund yield/volatility ranges, so testing it on arbitrary stock-price data would produce meaningless output. Everything here stays inside the yield-curve domain the model was actually trained on.

## Model — FinBERT-GF

- Hybrid architecture: ProsusAI/FinBERT contextual embeddings (attention pooling) + a Loughran-McDonald lexical branch (with skip connection) + a temporal Bund-yield branch, combined via a gated fusion layer.
- Trained on ECB speeches (1999–2026) merged with German 10Y Bund yield data via the ECB SDW API.
- Labels (Dovish / Neutral / Hawkish) are derived from 2-day forward yield changes using 35th/65th percentile thresholds.
- Full training pipeline, ablations, and results are documented in the accompanying thesis.

## How it works

1. A simulated yield tape moves tick-by-tick on a randomized walk scaled to realistic Bund volatility.
2. A pool of ECB-style headlines fires periodically and nudges the tape's drift (hawkish headlines push yields up, dovish headlines pull them down).
3. Every few ticks, the **actual trained FinBERT-GF checkpoint** classifies the current headline (plus live temporal features) and logs its read.
4. The user trades manually — Buy positions for a "yields up" view, Sell for a "yields down" view — using a fraction-of-portfolio slider.
5. A live candlestick chart, trade log, and running P&L update in real time.

## Tech stack

- **Model**: PyTorch, HuggingFace Transformers (ProsusAI/FinBERT)
- **Interface**: Gradio, Plotly
- **Environment**: Google Colab (GPU)

## Running it

1. Open the notebook in Google Colab.
2. Mount Google Drive (the model checkpoint is loaded from there — see note below).
3. Run all cells top to bottom. The final cell launches the Gradio app and prints a public link.

**Note on the model checkpoint**: `best_model.pt` is not included in this repo (model weights are too large for GitHub). It's loaded from Google Drive in the notebook — set `CHECKPOINT_PATH` to wherever you've stored your own copy, or contact me for access.

## Related work

This project builds directly on my MS thesis, *"[Thesis title: Quantify ECB Speeches and its influence on European Bund Market through signal extraction using DL and NLP]."* A separate portfolio project — a RAG-based ECB speech explorer using the same FinBERT-GF model — is also in progress.

## Disclaimer

This is a research/portfolio demo. It does not use real market data feeds, does not involve real money, and is not intended as financial advice or a production trading system.

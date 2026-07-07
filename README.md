# ai-catchup — 完全無料・完全ローカルの「AI業界レーダー」

**19の無料ソースを並列巡回し、埋め込みナレッジと機械シグナルで選別、ローカルLLMが「今、最速で有益な10件」を根拠付きで届ける。API課金ゼロ。クラウド送信ゼロ。**

```
*1. 🚀 New Realtime models (GPT-realtime-2.1) on the API*
   `HN · OpenAI` — 既知情報との重複なし / LLM評価 9/10 / リアルタイム処理能力の向上は応用範囲を大きく広げるため
*2. 📄 LLM-as-a-Verifier: A General-Purpose Verification Framework*
   `arXiv cs.AI, HF Daily Papers` — 2ソースが同時報道 / LLM評価 7/10 / 検証を新たなスケーリング軸とした点が注目
```

## なぜ作ったか

AIニュースは速すぎ、多すぎる。X・RSS・arXivを毎朝追うのは仕事にならないし、
キュレーションSaaSは月額課金かAPI課金が付いてくる。

**このツールの答え: 選別ロジックを全部手元で動かす。**

- 収集は無料の公開フィード/APIだけ(キー登録すら不要)
- 推論は[Ollama](https://ollama.com)のローカルモデル(既定: `gemma4:e4b` + `embeddinggemma`)
- 蓄積はSQLite。**運用コストは電気代のみ**

## 仕組み — LLMの主観に任せない設計

LangGraphのステートグラフとして実装。「今これが重要」の判定は、
まず**決定的な機械シグナル**で行い、LLMは最終スコアと根拠文だけを担当する:

```
collect ─→ dedup ─→ embed_store ─→ signals ─→ judge ─→ rank ─→ digest ─→ deliver
 19ソース   SQLite    埋め込み+蓄積   ↓            gemma4    ストーリー   Slack /
 並列fetch            (90日ナレッジ)              e4b採点    重複統合    stdout
                                    novelty      (JSON強制)
                                    corroboration
                                    tier / recency
```

| シグナル | 意味 |
|---|---|
| **corroboration** | 複数の独立ソースが数時間内に同じ話題を報道 = 最強の速報シグナル(埋め込みコサイン類似で story クラスタリング) |
| **novelty** | 過去90日のナレッジベースとのベクトル距離。「見たことのない話題」が浮上する |
| **tier** | 公式ラボ(tier 1) > キュレーション(2) > コミュニティ(3) |
| **recency** | 24時間半減期の指数減衰 |

最終スコア = LLM評価 55% + 機械シグナル 45%。LLMがハルシネーションしても
ランキングを乗っ取れない。同一ストーリーがN経路で届いた場合は1件に統合し、
`arXiv cs.AI, HF Daily Papers` のように出典を併記する。

## ソース(すべて無料・認証不要)

公式ラボRSS(OpenAI/DeepMind/NVIDIA/HF)、HN Algolia新着×4クエリ、
Reddit(r/LocalLLaMA, r/MachineLearning)、arXiv cs.AI、Techmeme、TechCrunch AI、
Publickey(日本語)、**HF Daily Papers**(コミュニティ投票済み論文)、
**Bluesky公開検索**×3クエリ(研究者の一次発信)、**GitHub Search API**(直近7日に
生まれて急伸中のAIリポジトリ)。

1ソースの死活がパイプラインを止めない fail-soft 設計。死んだフィードは
ログに残して先へ進む。

## クイックスタート

```bash
# 1. モデル準備(初回のみ、無料)
ollama pull gemma4:e4b && ollama pull embeddinggemma

# 2. インストール
git clone https://github.com/kento-cell/ai-catchup.git
cd ai-catchup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. 実行(まずはドライラン)
aicatchup run --dry-run
```

Slackに届けたい場合は `.env` に webhook を1行:

```bash
cp .env.example .env   # SLACK_WEBHOOK_URL=https://hooks.slack.com/...
aicatchup run
```

cronで朝夕2回など好きな周期で:

```cron
0 8,20 * * * cd /path/to/ai-catchup && .venv/bin/aicatchup run
```

## 実測

M2 Pro (16GB) で **954件収集 → 400件埋め込み → 30件LLM審査 → 10件配信を約1分40秒**。
GPU不要(Apple SiliconはMetal自動使用)。モデルダウンロード後はオフラインで動く。

## カスタマイズ

- ソース追加: `aicatchup/sources.py` の `SOURCES` にタプルを1行足すだけ
- モデル変更: `.env` の `CATCHUP_LLM_MODEL`(頭脳)/ `CATCHUP_EMBED_MODEL`(埋め込み)
- 配信件数/審査数/言語: `CATCHUP_TOP_N` / `CATCHUP_JUDGE_CANDIDATES` / `CATCHUP_LANG=ja|en`

## English TL;DR

Zero-cost, local-first AI news radar. 19 free sources fetched in parallel,
deduplicated, embedded into a SQLite knowledge base (Ollama embeddings),
scored by deterministic signals — cross-source corroboration, novelty vs.
90 days of accumulated knowledge, source tier, recency — then judged by a
local LLM against a strict JSON rubric. Final ranking blends LLM 55% /
signals 45%, so a hallucinating judge can never hijack it. Same story via
N sources collapses into one entry with merged attribution. Slack or stdout.
~100 seconds end-to-end on an M2 Pro. No API keys, no cloud, no running cost.

## License

MIT

# TinyLLaVA_Factory — Repo Findings

> Code-verified against the current checkout. Line numbers are exact as of
> investigation. Task 1 (data schema) was produced by the `repo-investigator`
> task; Tasks 2–6 (tower/connector interface, CLI args, registration mechanics)
> and the BulkFormer contract were verified by follow-up investigation and are
> filled in below (§5–§9).
>
> **Confirmed project decisions this spec assumes:** live frozen forward (the
> tower wraps the real BulkFormer model, no precomputed embeddings);
> **BulkFormer-127M** (`dim=640`) → per-sample embedding `dim+3 = 643` →
> `vision_hidden_size = 643`; encoder **frozen** throughout; LLM backbone left as
> the existing placeholder (out of scope).
>
> **⚠ SUPERSEDED 2026-08-13 — encoder scale.** The project has locked
> **BulkFormer-93M** (`dim=512`) → embedding `515` → `vision_hidden_size = 515`.
> Every `127M` / `640` / `643` figure below is therefore historical; the
> mechanism it describes is unchanged, only the width. See `llm_training_plan.md`
> §2 for the lock and its justification. (The LLM backbone placeholder was also
> since resolved to Vicuna-7B — see `integration/vicuna_smoke_test_result.md`.)

## 1. Confirmed training-sample JSON schema

The training data file is a single JSON **array** of sample objects, loaded
whole (not JSONL):

- `tinyllava/data/dataset.py:29` — `list_data_dict = json.load(open(data_path, "r"))`

Each sample object uses these **top-level fields**:

| Field | Required? | Where read | Meaning |
|---|---|---|---|
| `conversations` | **Yes** | `dataset.py:59` `sources["conversations"]` | List of turn dicts (the training text). Accessed with `[...]` bracket → mandatory. |
| `image` | Optional | `dataset.py:60` `if 'image' in sources:` then `dataset.py:61` `self.list_data_dict[i]['image']` | Relative image path. Presence toggles image loading; absence + `is_multimodal` yields a zero image tensor (`dataset.py:66-70`). |
| `id` | **Not used by loader** | — | Not referenced anywhere in `tinyllava/data/`. If present it is simply ignored. |

No other top-level field is read by the data pipeline.

### Conversation / turn structure

Each element of `conversations` is a dict with exactly two fields — `from` and `value`:

- `tinyllava/data/template/base.py:54` — `if i == 0 and message['from'] != 'human':`
- `tinyllava/data/template/base.py:58` — `question_list.append(message['value'])`
- `tinyllava/data/template/base.py:60` — `answer_list.append(message['value'])`

Turn parsing logic (`base.py:46-64`, `_get_list_from_message`):
- Roles are split by **alternation**, not by reading every role string. Only the
  literal `'human'` is checked, and only for turn index 0 (to detect a leading
  non-human/system turn and skip it).
- After that, even-index turns → questions, odd-index → answers (offset by
  `first_is_not_question`).
- The string `'gpt'` is **never** checked in code — the assistant role is
  inferred purely by position. The answer turn's `from` value is conventionally
  `"gpt"` but the loader does not validate it.
- `base.py:62-63` asserts `len(question_list) == len(answer_list)`, i.e. turns
  must pair into (human, gpt) rounds. Multi-round conversations are supported.

### Image-token placement

- The literal `<image>` token (value of `DEFAULT_IMAGE_TOKEN`) is expected to
  appear **inside a human turn's `value`** string. In `base.py:86-88`:
  ```python
  if DEFAULT_IMAGE_TOKEN in question:
      question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
      question = self.format_image_token.apply(content=question).strip()
  ```
  The token is stripped from wherever it sits and re-inserted in a canonical
  position by each template's `format_image_token` slot (e.g.
  `"<image>\n{{content}}"`, `llama_template.py:22`). So placement within the
  human turn does not matter — the template normalizes it to the front.
- Convention in LLaVA-style data: put `<image>\n` at the start of the first
  human turn's `value`.

### Representative example structure (verified against the fields the code reads)

```json
[
  {
    "id": "unique-sample-id",
    "image": "subdir/000000123.jpg",
    "conversations": [
      { "from": "human", "value": "<image>\nWhat is in this image?" },
      { "from": "gpt",   "value": "A cat sitting on a mat." }
    ]
  },
  {
    "id": "text-only-sample",
    "conversations": [
      { "from": "human", "value": "Hello, who are you?" },
      { "from": "gpt",   "value": "I am an assistant." }
    ]
  }
]
```

**Pretrain template special case:** its `format_user` slot is literally
`"<image>"` and `format_image_token` is empty (`pretrain_template.py:19-20`), so
pretrain samples' human turns are effectively replaced by the image token.

## 2. Hardcoded `image` / `<image>` / `DEFAULT_IMAGE_TOKEN` locations

### `DEFAULT_IMAGE_TOKEN` definition
- `tinyllava/utils/constants.py:9` — `DEFAULT_IMAGE_TOKEN = "<image>"`

### Literal `"<image>"` string hardcoded (all would need changing if token renamed)

Inside the data/training pipeline:
- `tinyllava/data/template/base.py:140` — `prompt.split('<image>')` in
  `tokenizer_image_token`. **Load-bearing:** the prompt is split on the literal
  `'<image>'` to locate where to inject `IMAGE_TOKEN_INDEX` (-200). Uses the raw
  literal, NOT the constant, so it must be kept in sync with `DEFAULT_IMAGE_TOKEN`.
- `tinyllava/data/template/pretrain_template.py:20` — `format_user = EmptyFormatter(slot="<image>")`
- `tinyllava/data/template/pretrain_template.py:27` — `mask_len = len(self.tokenizer_image_token("<image>", tokenizer))`
- `tinyllava/data/template/llama_template.py:22` — `format_image_token = StringFormatter(slot="<image>\n{{content}}")`
- `tinyllava/data/template/phi_template.py:17` — same slot pattern
- `tinyllava/data/template/qwen2_instruct_template.py:17` — same
- `tinyllava/data/template/qwen2_base_template.py:17` — same
- `tinyllava/data/template/gemma_template.py:20` — same

### `DEFAULT_IMAGE_TOKEN` constant used (data pipeline)
- `tinyllava/data/template/base.py:86` — `if DEFAULT_IMAGE_TOKEN in question:`
- `tinyllava/data/template/base.py:87` — `question.replace(DEFAULT_IMAGE_TOKEN, '')`

### Literal `<image>` outside the training data pipeline (inference/eval/serve only)
- `tinyllava/utils/message.py:60`; `tinyllava/eval/model_vqa_science.py:46,53`;
  `tinyllava/eval/eval_science_qa.py:85`; `tinyllava/eval/model_vqa_mmmu.py:111`;
  `tinyllava/serve/app.py:72-74`
- `DEFAULT_IMAGE_TOKEN` used in: `tinyllava/eval/model_vqa_pope.py:47`,
  `model_vqa_loader.py:47`, `run_tiny_llava.py:53`, `model_vqa.py:49`,
  `serve/cli.py:72`

### Literal top-level `'image'` JSON key hardcoded in the data loader
- `tinyllava/data/dataset.py:44` — `img_tokens = 128 if 'image' in sample else 0` (`lengths`)
- `tinyllava/data/dataset.py:53` — `cur_len = cur_len if 'image' in sample else -cur_len` (`modality_lengths`)
- `tinyllava/data/dataset.py:60` — `if 'image' in sources:` (image-loading gate)
- `tinyllava/data/dataset.py:61` — `image_file = self.list_data_dict[i]['image']` (reads path)
- `tinyllava/data/dataset.py:65` — `data_dict['image'] = image` (**internal** output-dict key)
- `tinyllava/data/dataset.py:70` — `data_dict['image'] = torch.zeros(...)` (**internal** output key)
- `tinyllava/data/dataset.py:109` — `if 'image' in instances[0]:` (collator; **internal** key)
- `tinyllava/data/dataset.py:110` — `images = [instance['image'] ...]` (collator; **internal** key)

Dual use: lines 44/53/60/61 read the **JSON input field** `image`; lines
65/70/109/110 read/write the **internal batch dict key** `image`. Renaming the
JSON field means changing 44, 53, 60, 61 only (internal key at 65/70/109/110 can stay).

### Literal `conversations` / `from` / `value` field names hardcoded
- `conversations`: `dataset.py:45`, `dataset.py:52`, `dataset.py:59`
- `from`: `template/base.py:54`
- `value`: `template/base.py:58`, `template/base.py:60`, `dataset.py:45`, `dataset.py:52` (`conv['value']`)
- role literal `'human'`: `template/base.py:54` (only hardcoded role check; `'gpt'` never hardcoded)

## 3. Image folder / file path / extension conventions

- **`--image_folder` equivalent:** `data_args.image_folder`
  (`tinyllava/utils/arguments.py:43` — `DataArguments.image_folder: Optional[str] = field(default=None)`).
- **Path join:** `tinyllava/data/dataset.py:62-63`:
  ```python
  image_folder = self.data_args.image_folder
  image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
  ```
  The sample's `image` field is a **path relative to `image_folder`**. It may
  include subdirectories (plain `os.path.join`, so nested paths like
  `"coco/train2017/xxx.jpg"` work).
- **Extensions/format:** No extension is hardcoded or assumed. The `image` field
  must carry the full filename including extension. Loading is
  `PIL.Image.open(...).convert('RGB')` — any PIL-decodable format (jpg, png,
  jpeg, ...) is accepted and converted to 3-channel RGB.
- **Truncated images tolerated:** `ImageFile.LOAD_TRUNCATED_IMAGES = True` at
  `dataset.py:20` and `image_preprocess.py:9`.
- **Aspect-ratio handling** (`image_preprocess.py:19-26`, via
  `data_args.image_aspect_ratio`, default `'square'` per `arguments.py:44`):
  `'pad'` → pad to square with mean-color background; `'anyres'` → multi-res
  patching; otherwise → straight to the HF `image_processor`.
- **Missing-image behavior:** no `image` field + `data_args.is_multimodal` True
  (`arguments.py:42`, default True) → zero tensor of shape
  `(3, crop_height, crop_width)` substituted (`dataset.py:66-70`).

## 4. Unresolved / couldn't confirm from code

- **`id` field:** No sample `.json` training file exists in this checkout to
  inspect a real example (only Python source under `tinyllava/data/`). The `id`
  field is conventional in LLaVA data but is **provably unused** by this loader
  (absent from all grep results). Its exact naming/format can't be confirmed
  from code because nothing reads it.
- **`--image_folder` CLI flag literal:** `DataArguments` uses HF
  `HfArgumentParser`, so field `image_folder` maps to `--image_folder`. The
  dataclass field name is confirmed (`arguments.py:43`); the exact CLI string in
  the training entrypoint/shell scripts was not opened here — inferred from HF
  argparse convention.
- **Assistant role string:** Convention is `"gpt"` for the assistant `from`
  value, but this is **not verified by any code check** — the loader infers
  assistant turns positionally (`base.py:46-64`) and never compares against
  `'gpt'`. Any string works as long as turns alternate human/assistant.
- **First-turn system role handling:** `base.py:52-55` skips a first turn whose
  `from != 'human'` (`first_is_not_question`), implying an optional leading
  system/non-human turn is tolerated, but no sample confirms the exact `from`
  value used.
- **`<image>` in non-first / multiple human turns:** `base.py:86` checks each
  question turn independently and `base.py:140` splits on every occurrence, so
  multiple/any-position image tokens are handled mechanically, but no data
  sample confirms the intended convention.

## 5. Vision tower interface (`tinyllava/model/vision_tower/base.py`)

`class VisionTower(nn.Module)` (line 15).

- **`__init__(self, cfg)`** (16-20): `cfg` is `config.vision_config` (an HF
  `PretrainedConfig`), NOT the top-level `TinyLlavaConfig`. Must set
  `self._vision_tower = None`, `self._image_processor = None`, `self.config = cfg`.
- **`load_model(self, vision_tower_name, **kwargs)`** (23-25): calls `_load_model`
  then `self._vision_tower.requires_grad_(False)` (freeze).
- **`_load_model`** (30-43): assumes an HF `from_pretrained` model or a raw
  `pytorch_model.bin` state-dict. **Must be overridden** for BulkFormer.
- **`forward(self, x, **kwargs)`** (47-58): assumes `self._vision_tower(x,
  output_hidden_states=True)` → `.hidden_states`, selects
  `kwargs['vision_feature_layer']` (default -2), drops CLS for `'patch'`.
  Returns `(batch, num_patches, vision_hidden_size)`. **Must be overridden** for
  BulkFormer (no hidden_states, no CLS, expression input).
- Property `vision_tower` getter/setter (62-68).

**Construction** (`modeling_tinyllava.py:64`):
`VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)`.
**Runtime** (`encode_images`, `modeling_tinyllava.py:194-201`): sets
`vision_feature_layer` + `vision_feature_select_strategy` kwargs, moves images to
device/dtype, calls `self.vision_tower(images, **kwargs)` → `self.connector(...)`.
Concrete pattern: `clip.py:7-12`, `siglip.py:7-12`; override stub at
`siglip.py:15-20`.

### BulkFormerVisionTower — required implementation
- `__init__(self, cfg)`: read `getattr(cfg, 'bulkformer_variant', 'BulkFormer-127M')`;
  build BulkFormer (§8) into `self._vision_tower`; set `self._image_processor` to a
  pass-through shim exposing `crop_size`/`size` (only for the `dataset.py:69` zero
  fallback, which never fires on the transcriptomic path).
- Override `_load_model`/`load_model`: load the `.pt` ckpt (§8), then freeze.
- Override `forward(self, x, **kwargs)`: `x=[B,20010]` →
  `self._vision_tower(x, mask_prob=0.0, output_expr=False)` → `[B,20010,643]` →
  `.mean(dim=1)` → `.unsqueeze(1)` → **`[B, 1, 643]`** (one token/sample; 20,010
  tokens would overflow the LLM context).

## 6. Connector interface (`tinyllava/model/connector/base.py`)

`class Connector(nn.Module)` (line 7).
- `__init__(self, config=None)` (8-10): sets `self._connector = None`; receives
  full `TinyLlavaConfig`.
- `load_model` (12-23): optional `pretrained_connector_path/pytorch_model.bin`
  load, then freeze.
- `forward(self, x)` (28-29): `return self._connector(x)`. Input tower output
  `(B, num_patches, vision_hidden_size)`; output `(B, num_patches, hidden_size)`
  (LLM dim).

Stock `linear` connector already maps 643→LLM:
`nn.Linear(config.vision_hidden_size, config.hidden_size)` (`connector/linear.py:12-14`).
`mlp.py:25` parses `config.connector_type` (`mlp<d>x_gelu`) for depth.

### TranscriptLinearConnector
`@register_connector('transcript_linear')`, `self._connector =
nn.Linear(config.vision_hidden_size, config.hidden_size)`. Functionally identical
to `linear`; named for extensibility. Reusing `--connector_type linear` is a
valid no-new-code alternative.

## 7. Registration + import chain (the critical integration point)

- Registry `VISION_TOWER_FACTORY = {}` (`vision_tower/__init__.py:6`);
  `register_vision_tower(name)` (18-24) — **silent no-op on name collision**
  (returns existing class, 20-21). `VisionTowerFactory(name)` (8-15): takes
  `name.split(':')[0]`, then **substring** match `registered.lower() in that.lower()`
  (12). Connector registry mirrors this (`connector/__init__.py`), no `:`-split.
- **Auto-import:** `vision_tower/__init__.py:27-29` runs
  `import_modules(dir, "tinyllava.model.vision_tower")`; `utils/import_module.py:7-9`
  imports every `*.py` **not** starting with `_`/`.`. Chain: entrypoints do
  `from tinyllava.model import *` (`train/train.py:11`, `custom_finetune.py:7`) →
  `model/__init__.py:2-3` → the two `import_modules` calls. **So dropping
  `bulkformer.py` into `tinyllava/model/vision_tower/` auto-registers it — no
  `__init__.py` edit.**
- **Pitfalls:** filename must not start with `_`/`.`; the registered name must be a
  substring of the `--vision_tower` value (or use `name:path` prefix); collisions
  silently no-op. Use a distinctive name (`bulkformer`).
- Invoke as `--vision_tower bulkformer:<config-dir>` → prefix `bulkformer` selects
  the class; suffix `<config-dir>` becomes `vision_config.model_name_or_path`
  (`configuration_tinyllava.py:124,129`).

## 8. BulkFormer model contract (from `linear_probe/extract.py`, verified)

- **Construct** (`extract.py:227-245`): `sys.path.insert(0, bulkencoders/BulkFormer)`;
  load graph `SparseTensor(row, col, value)` from
  `checkpoints/bulkformer/support/{G_tcga.pt, G_tcga_weight.pt}`, `gene_emb` from
  `support/esm2_feature_concat.pt`; then `BulkFormer(dim, p_repeat, graph=graph,
  gene_emb=gene_emb, bins=0, gb_repeat=1, bin_head=12, full_head=8, gene_length=20010)`.
- **Variant table** (`extract.py:68-74`): 37M(128,1), 50M(256,2), 93M(512,6),
  **127M(dim=640, p_repeat=8, `BulkFormer-127M.pt`)**, 147M(640,12).
- **Checkpoint load** (`extract.py:219-224`): `torch.load(path, map_location="cpu",
  weights_only=False)`, strip `module.` prefix, `load_state_dict(strict=True)`.
  Path `bulkencoders/checkpoints/bulkformer/models/BulkFormer-127M.pt`.
- **Input** (`extract.py:149-183`): raw counts → gene-length TPM → `log1p` → reorder
  to 20,010-gene ENSG vocab (`support/bulkformer_gene_info.csv`), missing genes =
  `-10.0`. Tensor `x=[B,20010]`.
- **Forward** (`extract.py:271-273`): `model(x, mask_prob, output_expr=False)` →
  `[B,20010,dim+3]`; `mean(dim=1)` → `[B,dim+3]`. 127M → **643**.
- **mask_prob:** CVD corpus H5 provides all 20,010 vocab genes → `mask_prob = 0.0`
  (per `linear_probe/writeup.md` §1). Default `0.0`.
- All 5 checkpoints + support tensors present on disk. **CPU-only** (GCNConv uses
  `torch_sparse` CPU kernels; MPS crashes); 127M ≈ 3 s/sample CPU.

## 9. CLI + training-stage reference (confirmed)

Args (`tinyllava/utils/arguments.py`, parsed in `train/train.py:52-54`,
`custom_finetune.py:22-24`): `--model_name_or_path`(13)=LLM; `--vision_tower`(16)=tower
(`train.py:38` `split(':')[-1]`); `--vision_tower2`(17)=MoF second tower;
`--connector_type`(18); `--conv_version`(45); `--training_recipe`(50)
common/lora/qlora; `--tune_type_llm`(51) frozen/full/lora/qlora_int*;
`--tune_type_vision_tower`(52) frozen/full/partially-tune; `--tune_type_connector`(54)
frozen/full; `--pretrained_model_path`(79) — presence distinguishes stage 2.
Applied by `training_recipe/base.py:32-87`.

**Two stages** (sequential in `train_phi.sh:16-17`):
- **Stage 1 pretrain** (`pretrain.sh:39-42`): `--tune_type_llm frozen
  --tune_type_vision_tower frozen --tune_type_connector full`, `--conv_version
  pretrain`, LR 1e-3 → **connector only**.
- **Stage 2 finetune** (`finetune.sh:41-46`): `--tune_type_llm full
  --tune_type_vision_tower frozen --tune_type_connector full`, `--conv_version
  $CONV_VERSION`, `--pretrained_model_path .../-pretrain`, LR 2e-5 → **LLM +
  connector, tower frozen**.
- Both keep the tower **frozen** by default → matches our decision, no extra flags.
  Both invoke `train.py` via `deepspeed --zero3`.

For BulkFormer, JSON `"image"` points to a `.npy` under `--image_folder`.

## 10. Unresolved / requires edits OUTSIDE the integrator's declared file scope

These are real and cannot be worked around inside tower/connector files alone.
They are handled as preceding edits (not by the integrator agent):

1. **Data loader (`dataset.py:60-70`)** hardcodes `Image.open(...).convert('RGB')`
   + HF image processor (§1, §3). A **transcriptomic branch** is required: when the
   `"image"` value ends in `.npy`, load `torch.from_numpy(np.load(path)).float()`
   (a `[20010]` vector), skip `image_preprocess`, and guard the no-image fallback
   (69-70) so it does not force `[3,H,W]`. The collator stacks equal-length
   `[20010]` vectors into `[B,20010]` unchanged.
2. **Vision-config loader (`configuration_tinyllava.py:124`)** calls
   `AutoConfig.from_pretrained(vision_model_name_or_path.split(':')[-1])`;
   BulkFormer has no HF-hub config. Resolved by a **local minimal HF config dir**
   `integration/bulkformer_hf_config/config.json` (`model_type: clip_vision_model`,
   `hidden_size: 643`, `bulkformer_variant`), passed as
   `--vision_tower bulkformer:<abs path>`. Makes lines 124-131 populate
   `vision_hidden_size=643` + `vision_config.model_name_or_path` without editing
   `configuration_tinyllava.py`. If AutoConfig drops `bulkformer_variant`, default
   the variant in the tower.

Original Task-1 unresolved items (sample `id`, exact `--image_folder` CLI string,
assistant role string) remain as noted in §4 and do not affect BulkFormer wiring.

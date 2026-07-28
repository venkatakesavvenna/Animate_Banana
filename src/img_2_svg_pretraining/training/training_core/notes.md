### DOUBTS
1. `image_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)`
    ```
    The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
    ```
    - should we use `use_fast = False` in `_make_processor` in `data_modules/qwen/qwen_data.py`??
1. should we use SEG as a special token?? --> It does not affect the training even 1 bit.. it only affect tokenizing and normalizing or decoding.
    
1. `label = torch.ones(masks[0].shape[0], masks[0].shape[1]) * ignore_label` what does this do in layout_utils.py:400?? (Also, `ingore_label=255` Why??) Can be removed.
1. `return_assistant_tokens_mask`?? (https://gemini.google.com/share/771ba1fc23d6)
1. `sam.eval()` is used to freeze everything and then we manually make the `p.requires_grad=True` but this is wrong.. since `eval` mode has other side effects like disabling dropout and maybe few other things.. we might not know about.
 - we have to put in `.train` mode explicitly
1. `get_visual_embs()` in `models/dam/sam_model.py` is running each image.. individually.. but we can run them at once.. in a batch.
1. SAM_MODEL
    - in sam.forward why do we need sparse_emb = to(vlm_hidden_states.dtype)
    - `cur_hidden_states.shape[1]` must be equal to `prompt_embed_dim` in `_build_sam()`
    - instead of sending label_list and resize_list.. can we calculate in sam.forward itself??
    - do we need to send original_images?? will they be loaded on to gpu?? should we load directly during val rather than sending in from data collator itself?
1. models/qwen/qwen_model.py
    - self.backbone.qwen.config.use_cache = False... WHY???
1. QWEN_SAM
    - `out_dim = self.sam_head.sam.prompt_encoder.embed_dim` this was used instead of out_dim  = config.out_dim
    - dropout of fcs layer = 0??
1. Verify the tensor dimension of sam_model.py, clearly. very ambiguous right now.
 - what is the Pad_Value in compute_metrics.. what to use exactly??
1. save sam32?? is it even requried?? the model is training in bf16 only by default..


### Possible Problems
1. system\nYou are a helpful assistant\nuser\n\nPlease give the layout of the document\nassistant\n<l>section-header</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>section-header</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>section-header</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>section-header</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>picture</l>___ [SEG] ___ <l>caption</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>section-header</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>picture</l>___ [SEG] ___ <l>caption</l>___ [SEG] ___ <l>text</l>___ [SEG] ___ <l>list-item</l>___ [SEG] ___ <l>list-item</l>___ [SEG] ___ <l>page-footer</l>___ [SEG] ___ \n -> system prompt missing.
2. 
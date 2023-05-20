from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, AutoConfig


def load_model():
    model_path = "/data/home/sunwubin/baihai_20230511"
    tokenizer = LlamaTokenizer.from_pretrained(model_path, fast_tokenizer=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto',
                                             from_tf=bool('.ckpt' in model_path),
                                             config=model_config).half()
    return model, tokenizer


model, tokenizer = load_model()
# model = model.to(device)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt = "今天天气很好"
gen_text = generator(prompt, max_length=150, num_return_sequences=4, num_beams=5)

for text in gen_text:
    print(text['generated_text'])
    print("\n=====\n")

from transformers import BloomTokenizerFast, BloomForCausalLM, GenerationConfig

tokenizer = BloomTokenizerFast.from_pretrained('./bloom-2b5-zh')
model = BloomForCausalLM.from_pretrained('./bloom-2b5-zh', device_map="auto",)
generation_config = GenerationConfig(temperature=0.1,
                                     max_length=100,
                                     top_p=0.98,
                                     top_k=10,
                                     do_sample=True,
                                     num_beams=1,
                                     no_repeat_ngram_size=2,
                                     early_stopping=True, num_return_sequences=2)
inputs = tokenizer.encode('Human: 我感冒了应该怎么办\nAssistant: ', return_tensors='pt')
inputs = inputs.to('cuda')
output = tokenizer.batch_decode(model.generate(input_ids=inputs, generation_config=generation_config,max_new_tokens=300))
print(output)

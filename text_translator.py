from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")


while True:
    # 2. Take input from user
    text = input("\nEnter an English sentence to translate (or type 'exit' to quit):\n> ")
    if text.lower() == "exit":
        break
    #text = "The weather is nice today."
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)


    # 4. Show result
    print("🇫🇷 Translated:", output)

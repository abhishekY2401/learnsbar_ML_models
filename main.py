from flask import Flask
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

sentences = [
    "Dr. Cooper? Yes. This is about Mrs. Lakewood. I'm concerned that she's having difficulty breathing and her blood pressure is dropping. Yes, sir. And she's a 65-year-old woman, two days post-op. She has medical problems of obesity, cardiac disease, and diabetes. And while I'm concerned that right now she may be actually going into some cardiac event yes, sir, I'd like you to come down right now. Are you on your way? Good. Thanks.",
    "Hi, Doctor. This is about Mrs. Lakewood. Postop surgery a couple of days ago. She's her family is giving me a lot of trouble. They're really high maintenance. I've been talking to them quite a bit. Anyway, something's going on. Maybe. It's not important. Lab work came back, though. Okay. We did an IV on her because it didn't seem to help the urine output. The oxygen saturation is okay when she puts it on, which is about half the time anyway. The incision is beginning to bug her, and she's using a lot of morphine, which makes it hard to assess what's really going on. I don't know. What do you think we ought to do?",
    "Hello, Doctor. The subject here is Mrs. Lakewood. Postop surgery was performed just recently. She and her family are causing me lots of problems. They require a lot of upkeep. I've been chatting with them a lot. In any case, something is happening. Maybe. It doesn't matter. However the lab results were returned. Okay. She had an IV because the urine production didn't seem to improve. When she puts it on, which is roughly half the time, the oxygen saturation is fine. It's challenging to determine the true nature of her problems because the incision is starting to bother her and she's taking a lot of morphine. I'm not sure. What should we do, in your opinion?",
    "Doctor Cooper? Yes. The subject here is Mrs. Lakewood. She's having trouble breathing, and her blood pressure is decreasing, which worries me. Sure, sir. A 65-year-old woman, she is two days post-op. She suffers from diabetes, heart disease, and obesity as medical conditions. And yes, sir, I would like you to come down right away even though I'm worried that she might actually be having a cardiac attack right now. Ahead of schedule? Good. Thanks."
]

example_sentence = "Dr. Cooper? Yes. This is about Mrs. Lakewood. I'm concerned that she's having difficulty breathing and her blood pressure is dropping. Yes, sir. And she's a 65-year-old woman, two days post-op. She has medical problems of obesity, cardiac disease, and diabetes. And while I'm concerned that right now she may be actually going into some cardiac event yes, sir, I'd like you to come down right now. Are you on your way? Good. Thanks."


@app.route("/")
def homepage():
    return "Hello World"


@app.route("/cosine")
def cosine_similar():
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')

    example_tokens = tokenizer.encode_plus(
        example_sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    print(example_tokens.keys())

    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(
            sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # restructure a list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    resized_attention_mask = attention_mask.unsqueeze(
        -1).expand(embeddings.size()).float()

    masked_embedding = embeddings * resized_attention_mask

    summed_masked_embeddings = torch.sum(masked_embedding, 1)

    count_of_one_in_mask_tensor = torch.clamp(
        resized_attention_mask.sum(1), min=1e-9)

    mean_pooled = summed_masked_embeddings / count_of_one_in_mask_tensor

    mean_pooled = mean_pooled.detach().numpy()

    similar = cosine_similarity([mean_pooled[0]], mean_pooled[1:])
    return f'cosine_similarity: {similar}'


if __name__ == '__main__':
    app.run(debug=True)

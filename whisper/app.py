import os
import whisper
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from happytransformer import HappyTextToText, TTSettings

model = whisper.load_model("small")
exemplar_sentence = "The patient's name is Mrs. Helena Jones, a 65-year-old woman who was admitted to the hospital six days ago following a road traffic accident. She sustained multiple rib fractures and is currently being managed conservatively for her pain and closely monitored for her vital signs. The decision was made not to perform surgery on her. During the night, Mrs. Jones experienced multiple episodes of shortness of breath, particularly when repositioning in bed. In the morning, when the nurse greeted her at the bedside, she appeared relaxed and her oxygen saturation levels were above 96%. Two hours later, the nurse noticed that Mrs. Jones was becoming quite restless in bed and decided to do a full set of observations. The nurse found that her oxygen levels had significantly dropped, her blood pressure had also dropped, and she appeared very anxious. Upon checking her past medical history, the nurse noted that Mrs. Jones is usually hypertensive. Based on these findings, the nurse became concerned and decided to raise the issue using the SBAR (Situation, Background, Assessment, Recommendation) handover to the medical doctors who are also looking after Mrs. Jones."


def getScore(audio_url):

    result = model.transcribe(audio_url)
    print(result['text'])
    print(cosine_similar(result['text']))


def cosine_similar(sentence):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')

    example_tokens = tokenizer.encode_plus(
        sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    print(example_tokens.keys())

    tokens = {'input_ids': [], 'attention_mask': []}

    new_tokens = tokenizer.encode_plus(
        exemplar_sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
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

    eanm_pooled = mean_pooled.detach().numpy()

    similar = cosine_similarity([
                                 [0]], mean_pooled[1:])
    return f'cosine_similarity: {similar}'


audio_url = "./Demo.wav"

getScore(audio_url)

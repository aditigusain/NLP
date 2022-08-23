
# import third party modules --------------------------------------------------
import pandas
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# global variables ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# method definitions ----------------------------------------------------------
def chunks(list_of_elements, batch_size):
    """yield successive batch-sized chunks from list_of_elements"""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def evaluate_summaries_pegasus(dataset, 
                            metric, 
                            model, 
                            tokenizer, 
                            batch_size=16, 
                            device=device,
                            column_text="article",
                            column_summary="highlights"
                            ):
    article_batches = list(chunks(dataset[column_text], batch_size))
    target_batches = list(chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches),  total=len(article_batches)
        ):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                        padding="max_length",return_tensors="pt")
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device),
                                length_penalty=0.8,num_beams=8,max_length=128)

        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
                            for s in summaries]
        
        decoded_summaries = [d.replace("<n>"," ") for d in decoded_summaries]
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
    
    score = metric.compute()
    return score


dataset_samsum = load_dataset("samsum")
split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]

print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")
print(dataset_samsum["test"][0]["dialogue"])
print("\nSummary:")
print(dataset_samsum["test"][0]["summary"])


model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

score = evaluate_summaries_pegasus(
            test_sampled, 
            rouge_metric,
            model,
            tokenizer,
            batch_size=8
        )

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
print(pandas.Dataframe(rouge_dict, index=["pegasus"]))
   



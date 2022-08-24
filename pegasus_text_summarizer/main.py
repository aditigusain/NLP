
# import third party modules --------------------------------------------------
import pandas
import torch
from tqdm import tqdm
from datasets import load_dataset, load_metric
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
                            batch_size=4, 
                            device=device,
                            column_text="article",
                            column_summary="highlights"
                            ):
    article_batches = list(chunks(dataset[column_text], batch_size))
    target_batches = list(chunks(dataset[column_summary], batch_size))

    i=0
    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches),  total=len(article_batches)
        ):
        print(f"\n{i+1}th loop\n")
        inputs = tokenizer(article_batch, max_length=128, truncation=True,
                        padding="max_length",return_tensors="pt")
        print("recieved inputs")
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device),
                                length_penalty=0.8,num_beams=4,max_length=64)
        print("generated summaries")
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
                            for s in summaries]
        print("decoded summaries")
        decoded_summaries = [d.replace("<n>"," ") for d in decoded_summaries]
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
        print("metric prepared for computation")
        i=i+1
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

torch.cuda.empty_cache()
model_ckpt = "google/pegasus-cnn_dailymail"
print(f"\n\nmodel ckpt: {model_ckpt}")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print("\nTokenizer loaded")
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
print("\nModel loaded")
rouge_metric = load_metric("rouge")
print("\nmetric loaded")
score = evaluate_summaries_pegasus(
            dataset_samsum["test"], 
            rouge_metric,
            model,
            tokenizer,
            column_text="dialogue",
            column_summary="summary",
            batch_size=4
        )

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
print(pandas.DataFrame(rouge_dict, index=["pegasus"]))
   



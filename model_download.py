from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/t5-xl-ssm-nq"

tokenizer = AutoTokenizer.from_pretrained("google/t5-xl-ssm-nq")

model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-xl-ssm-nq")

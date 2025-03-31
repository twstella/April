import asyncio
import aiohttp
import pandas as pd
import glob
import os
import json
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import re
from openai import OpenAI

ollama_url = "http://localhost:11434/v1/chat/completions"
model_name = "deepseek-r1:latest"
semaphore = asyncio.Semaphore(3)
os.environ["OPENAI_API_KEY"]= "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"
client = OpenAI()

async def fetch(session, question, answer, idx, n_keywords):
	json_format={
	"format":{
		"type": "json_schema",
		"name": "keyword_and_category_extraction",
		"schema":{
			"type": "object",
			"properties":{
				"question_keywords":{
					"type":"array",
					"items":{
						"type": "string"
					},
				},
				"answer_keywords":{
					"type":"array",
					"items":{
						"type":"string"
					},
				},
				"category":{
					"type":"string"
				},
			},
			"required":["question_keywords","answer_keywords","category"],
			"additionalProperties":False
		},
		"strict": False
	}
	}
	# prompt = f"""
	# ## System Prompt:
	# You are a **keyword and category extraction assistant**.
	# You will be given a **question** and its **answer**.
	# Your task is to extract:
	# - A list of **{n_keywords} keywords from the question**
	# - A list of **{n_keywords +1} keywords from the answer**
	# - A **category** from the question
	# All extracted keywords and the category must be in **noun form only**.
	# ### Categories:
	# If the question relates to any of the following, assign exactly **one** category:
	# - **Definition**: Definition or explanation of a concept or object, label(Definition)
	# - **Conclusion**: Diagnosis, biomarker, biological pathway, medication, treatment, symptom, or cause of a disease, label(Conclusion)
	# - **Proposition of Fact**: A statement or question about verifiable truth, label(Fact)
	# - **Comparative Proposition**: Comparison between two or more entities, label(Compare)
	# - **Proposition of Value**: A question about point of view or value, label(Value)
	# - **Proposition of Policy**: A question about decision-making, method, application, or action, label(Policy)
	# - **Etc**: If the question statisfies none of the categories above
	# ### Output Format(JSON):
	# You must respond in the following **strict JSON format**:
	# ```json
	# {{
	# "question_keywords": ["keyword1","keyword2",...,"keyword{n_keywords}"],
	# "answer_keywords": ["keyword1","keyword2",...,"keyword{n_keywords+1}"],
	# "category": "Definition | Fact | Value | Compare | Policy | Conclusion"
	# }}
	# ```
	# """
	prompt = f"""
	## System Prompt:
	You are a **keyword and category extraction assistant**.
	You will be given a **question** and its **answer**.
	Your task is to extract:
	- A list of **{n_keywords} keywords from the question**
	- A list of **{n_keywords +1} keywords from the answer**
	- A **category** from the question
	All extracted keywords and the category must be in **noun form only**.
	### Categories:
	If the question relates to any of the following, assign exactly **one** category:
	- **Conclusion**: Diagnosis, biomarker, biological pathway, medication, treatment, symptom, or cause of a disease, label(Conclusion)
	- **Proposition of Fact**: A statement or question about verifiable truth, label(Fact)
	- **Comparative Proposition**: Comparison between two or more entities, label(Compare)
	- **Proposition of Value**: A question about point of view or value, label(Value)
	- **Proposition of Policy**: A question about decision-making, method, application, or action, label(Policy)
	- **Etc**: If the question statisfies none of the categories above
	### Output Format(JSON):
	You must respond in the following **strict JSON format**:
	```json
	{{
	"question_keywords": ["keyword1","keyword2",...,"keyword{n_keywords}"],
	"answer_keywords": ["keyword1","keyword2",...,"keyword{n_keywords+1}"],
	"category": "Fact | Value | Compare | Policy | Conclusion"
	}}
	```
	"""
	payload = {
		"model": model_name,
		"messages":[{"role":"user","content": prompt+question}]
	}
	headers = {"Content-Type": "application/json"}
	pattern = r"```json\s*({[\s\S]*?})\s*```"
	pattern2 = pattern2 = r"</think>\s*([\s\S]*)"
	async with semaphore:
		try:
			to_json = client.responses.create(model="gpt-4o-mini", input=[{"role":"system","content":prompt},{"role":"user","content":question}],text=json_format)
			final_chance= json.loads(to_json.output_text)
			print(final_chance)
			return idx, question, answer, final_chance["question_keywords"], final_chance["answer_keywords"], final_chance["category"]
			# # async with session.post(ollama_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=36000)) as response:
			# # 	if response.status==200:
			# # 		data = await response.text()
			# # 		data = json.loads(data)
			# # 		result = data["choices"][0]["message"]["content"]
			# # 		print("\n[result]")
			# # 		print(result)
			# # 		res_match = re.findall(pattern,result)
			# # 		if res_match:
			# # 			json_str = res_match[-1]
			# # 			parsed = json.loads(json_str)
			# # 			print("[JSON]")
			# # 			print(parsed)
			# # 			return idx, question, answer, parsed["question_keywords"], parsed["answer_keywords"], parsed["category"]
					
			# # 		else:
			# # 			try:
			# # 				match_str = re.search(pattern2, result)
							
			# # 				if match_str:
			# # 					print("\n[Second chance]")
			# # 					print(match_str.group(1))
			# # 					to_json = client.responses.create(model="gpt-4o-mini", input=[{"role":"system","content":"Extract question keywords, answer keywords, question category"},
			# # 					{"role":"user","content":match_str.group(1).strip()}],text=json_format)
			# # 					final_chance= json.loads(to_json.output_text)
			# # 					print(final_chance)
			# # 					return idx, question, answer, final_chance["question_keywords"], final_chance["answer_keywords"], final_chance["category"]
			# # 				else:
			# # 					print(f"[ERROR] {idx}:{question} -> STATUS: no JSON block")
			# # 					return idx, question, answer, [],[],"[ERROR]: no JSON block"
			# # 			except Exception as a:
			# # 				print(f"[ERROR]: Not able to translate to JSON format")
			# # 				return idx, question, answer, [], [], f"[ERROR] not able to translate to JSON format: {a}"
			# 	else:
			# 		print(f"[ERROR] {idx}:{question} -> STATUS: {response.status}")
			# 		return idx, question, answer,[],[], f"[ERROR]:{response.status}"
		except asyncio.TimeoutError:
			print(f"[Timeout] {idx}: {question}, {answer}")
			return idx, question, answer, [], [], "[ERROR]: Timeout Error"
		except Exception as e:
			return idx, question, answer, [], [], f"[ERROR]: {str(e)}"

async def process_file(file_path, n_keywords=3, batch_size=10000):
	print(f"Processing file: {file_path}")
	df = pd.read_csv(file_path)
	processed_rows = []
	async with aiohttp.ClientSession() as session:
		for idx, row in tqdm(df.iterrows(), total = len(df), desc="Processing rows"):
			question = row["question"]
			answer = row["cot"]
			try:
				result = await fetch(session, question, answer, idx, n_keywords)
				_, _, _, q_keywords, a_keywords, category = result
				new_row = row.copy()
				if len(q_keywords)==0 or len(a_keywords)==0 or (not category):
					new_row["question_keywords"] = ""
					new_row["answer_keywords"] =""
					new_row["question_category"]=""
				else:
					new_row["question_keywords"]= json.dumps(q_keywords)
					new_row["answer_keywords"]= json.dumps(a_keywords)
					new_row["question_category"] = category
				processed_rows.append(new_row)
				print(f"[UPDATED ROW {idx}] Category:{category}, Q keywords: {q_keywords}, A keywords: {a_keywords}")
			except Exception as e:
				print(f"[ERROR in row {idx}]: {str(e)}")
				new_row = row.copy()
				new_row["question_keywords"] = ""
				new_row["answer_keywords"] = ""
				new_row["question_category"]=""
				processed_rows.append(new_row)
			if (idx+1) % batch_size == 0 or idx == len(df) -1:
				if processed_rows:
					output_path = file_path.replace(".csv",f"{idx}_keywords.csv")
					temp_df = pd.DataFrame(processed_rows)
					temp_df.to_csv(
						output_path,
						mode='a' if os.path.exists(output_path) else 'w',
						header = not os.path.exists(output_path),
						index=False
					)
					processed_rows = []

def main():
	csv_files = glob.glob("sampled/*.csv")
	loop = asyncio.get_event_loop()
	for file_path in csv_files:
		loop.run_until_complete(process_file(file_path))

if __name__ == "__main__":
	main()


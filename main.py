import argparse
import os
import re
import numpy as np
import yaml
import openai
from pdfminer.high_level import extract_text

client = openai.OpenAI()

WC_MAX = int(4096 * 7 / 16)

def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text

def clean_extracted_text(text):
    text = text.replace('-\n', '')
    text = re.sub(r'\s+', ' ', text)
    return text

def create_prompt(text, total_num_req):
    return f"""英語の研究論文の一部を日本語で要約するタスクを行います。
研究論文は全部で{total_num_req + 1}個に分割しています。
以下のルールに従ってください。

・リスト形式で出力する (先頭は - を使う)
・簡潔に表現する
・不明な単語や人名と思われるものは英語のまま表示する

それでは開始します。

英語の論文の一部:
{text}

日本語で要約した文章:"""

def split_text(text, wc_max):
    words = text.split()
    chunks = [' '.join(words[i:i + wc_max]) for i in range(0, len(words), wc_max)]
    return chunks

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        pdf_files = [os.path.join(root, file) for file in files if file.endswith('.pdf')]
        total_files = len(pdf_files)
        for idx, file_path in enumerate(pdf_files):
            print(f'Processing {idx + 1}/{total_files}: {file_path}')
            process_pdf(file_path)

def get_all_pdf_files(directory):
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def process_pdf(file_path, total_files, current_idx):
    extracted_text = extract_text_from_pdf(file_path)
    clean_text = clean_extracted_text(extracted_text)

    splited_clean_text = split_text(clean_text, WC_MAX)
    summary = []

    for i, text_chunk in enumerate(splited_clean_text):
        attempt = 0
        while attempt < 3:  # 最大3回までリトライ
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "あなたは、プロの論文研究者で熟練した査読者です。あなたは、前後に問い合わせした内容を考慮して思慮深い回答をします。"},
                        {"role": "user", "content": create_prompt(text_chunk, len(splited_clean_text))}
                    ],
                    max_tokens=800,  # トークン数を減らす
                    n=1,
                    stop=None,
                    temperature=0.7,
                    top_p=1
                )

                if not response.choices:
                    raise ValueError("The API did not return a choice.")

                summary.append(response.choices[0].message.content)
                break  # 成功したらループを抜ける
            except Exception as e:  # すべての例外をキャッチ
                print(f"Error occurred: {str(e)}. Retrying...")
                attempt += 1
                time.sleep(2)  # エラー後の少しの休憩

    summary_text = '\n'.join(summary)
    print(summary_text)
    save_summary(file_path, summary_text)



def save_summary(file_path, summary_text):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = f'{base_name}_summary.txt'
    with open(summary_file_path, 'w') as f:
        f.write(summary_text)

def main():
    parser = argparse.ArgumentParser(description='Process all PDF files in a directory.')
    parser.add_argument('directory_path', help='The path to the directory containing PDF files.')
    args = parser.parse_args()

    pdf_files = get_all_pdf_files(args.directory_path)
    total_files = len(pdf_files)

    for idx, file_path in enumerate(pdf_files):
        print(f'Processing {idx + 1}/{total_files}: {file_path}')
        process_pdf(file_path, total_files, idx + 1)

if __name__ == "__main__":
    main()

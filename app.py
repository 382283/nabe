from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# 翻訳モデル（英語→日本語）
translator = pipeline("translation", model="staka/fugumt-en-ja")

# 要約モデル（日本語要約）
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# 長いテキストを分割する関数
def split_text(text, max_length=1024):
    # max_lengthより長いテキストを分割
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # 単語と空白の長さを計算
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            text = file.read().decode("utf-8")
            print(f"Uploaded text: {text}")

            # テキストが長い場合に分割
            text_chunks = split_text(text)

            # 翻訳を分割して行う
            translated_text = ""
            for chunk in text_chunks:
                try:
                    # チャンクの長さが十分でない場合はスキップ
                    if len(chunk.split()) < 5:  # 例えば、5単語未満なら翻訳しない
                        translated_text += chunk + " "
                    else:
                        # 翻訳を行う
                        translated_chunk = translator(chunk, max_length=1000)[0][
                            "translation_text"
                        ]
                        translated_text += translated_chunk + " "
                except Exception as e:
                    print(f"Error during translation: {e}")
                    translated_text += "[Translation Error] "

            # 要約
            try:
                # 要約用のmax_lengthをテキストに応じて調整
                if len(translated_text.split()) < 50:  # テキストが非常に短い場合
                    summary = translated_text  # 要約はスキップしてそのまま表示
                else:
                    summary_max_length = 130
                    summary = summarizer(
                        translated_text,
                        max_length=summary_max_length,
                        min_length=30,
                        do_sample=False,
                    )[0]["summary_text"]
            except Exception as e:
                print(f"Error during summarization: {e}")
                summary = "[Summarization Error]"

            return render_template(
                "result.html", translation=translated_text, summary=summary
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

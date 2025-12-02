from flask import Flask, render_template, request, url_for, jsonify
# from langchain_core.messages import HumanMessage, SystemMessage
# from agent import Agent
# from pydantic_ai import Agent, RunContext
from document_processing.process import RAGBot, get_prompt_template
from utils import load_credentials
import time
import os

from flask import send_from_directory
import uuid, os

app = Flask(__name__)
# agent = Agent('google-gla:gemini-2.5-flash')
config = {"configurable": {"thread_id": "abc123"}}
# messages = [SystemMessage(content=agent.system_prompts["default"])]
# create text file to store conversation
os.makedirs("./conversations/", exist_ok = True)
convo_txt = "./conversations/" + time.ctime().replace(' ', '_').replace(':', '_')

ragbot = RAGBot(source_dir="test_pages")
retriever, doc_store = ragbot.process_documents()
rag_chain = ragbot.build_bot(retriever, get_prompt_template())

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory("static/files", filename)

@app.route("/chat", methods=["POST"])
def get_tutor_response():
    user_msg = request.json['message']

    # Example: create a file as output
    response = ragbot.query(user_msg, rag_chain, get_images=True, doc_store=doc_store)
    file_id = str(uuid.uuid4())
    output_path = f"static/files/{file_id}.txt"
    with open(output_path, "w") as f:
        f.write("Hello! This is generated file content.")

    file_url = url_for("serve_file", filename=f"{file_id}.txt")

    return jsonify({
        "response": response["answer"],
        "file_url": file_url,
        "context": []
    })

    # normal RAG
    # response = ragbot.query(user_msg, rag_chain, get_images=False, doc_store=doc_store)
    # answer = response['answer']
    # if 'images' in response:
    #     print("images are thiere...")
    # else:
    #     print(response.keys())
    # with open(convo_txt, 'a') as f:
    #     f.write('User: ' + user_msg + '\n')
    #     f.write('Bot: ' + answer + '\n')
    # return jsonify({'response':answer})

    # multimodal:
    # response = ragbot.query(user_msg, rag_chain, get_images=True, doc_store=doc_store)
    # answer = response['answer']
    # if 'images' in response:
    #     print("images are thiere...")
    # else:
    #     print(response.keys())
    # with open(convo_txt, 'a') as f:
    #     f.write('User: ' + user_msg + '\n')
    #     f.write('Bot: ' + answer + '\n')
    # return jsonify({'response':answer})

    # # Build context list for frontend
    # context_list = []
    # if 'images' in response and response['images']:
    #     context_list.append({
    #         "text": answer,       # or clipped context text if available
    #         "images": response['images']
    #     })

    return jsonify({
        "response": response["answer"],
        "context": context_list
    })




if __name__ == "__main__":
    load_credentials()
    app.run(host='0.0.0.0', port=4999, debug=False)


"""
don't forget to export Google API Key and HF_TOKEN before running
"""
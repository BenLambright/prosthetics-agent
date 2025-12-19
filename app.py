from flask import Flask, render_template, request, url_for, jsonify
# from langchain_core.messages import HumanMessage, SystemMessage
# from agent import Agent
from pydantic_ai import Agent, RunContext
from document_processing.process import RAGBot, get_prompt_template
from utils import load_credentials
import time
import os
import shutil
import zipfile

from flask import send_from_directory
import uuid, os

from trimesh_tests import MeshScaler, calculate_scale

app = Flask(__name__)
# agent = Agent('google-gla:gemini-2.5-flash')
config = {"configurable": {"thread_id": "abc123"}}
# messages = [SystemMessage(content=agent.system_prompts["default"])]
# create text file to store conversation
os.makedirs("./conversations/", exist_ok = True)
convo_txt = "./conversations/" + time.ctime().replace(' ', '_').replace(':', '_')

ragbot = RAGBot(source_dir="pages")
retriever, doc_store = ragbot.process_documents()
rag_chain = ragbot.build_bot(retriever, get_prompt_template())

agent = Agent(
    'google-gla:gemini-2.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You are a friendly assistant. Determine if the user wants to query a document or create an STL file."
        "Answer 1 and only 1 if the question is specifically related to a diagram or assembly a 3D printed component"
        "Answer 2 and only 2 if they want to create an STL file."
        "Answer 3 and only 3 for any other question."
    ),
)

length_finder = Agent(
    'google-gla:gemini-2.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You are a helpful assistant that can determine the length of prosthetic components."
        "Given a description of a prosthetic by a user, return its length in millimeters as an integer."
        "Return only the integer value."
    ),
)

width_finder = Agent(
    'google-gla:gemini-2.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You are a helpful assistant that can determine the width of prosthetic components."
        "Given a description of a prosthetic by a user, return its width in millimeters as an integer."
        "Return only the integer value."
    ),
)

general_answer = Agent(
    'google-gla:gemini-2.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You are a helpful assistant for a 3D printing and prosthetics."
        "After answering the question, confirm if they want to ask about assembly or file geneeration, if so, please rephrae."
    ),
)

# @agent.tool  
# def query_doc(ctx: RunContext[str]) -> str:
def query_doc(ctx: str) -> str:
    """Query / RAG a document to get more information to answer the question about prosthetics or assembly."""
    # multimodal:
    response = ragbot.query(ctx, rag_chain, get_images=True, doc_store=doc_store)
    answer = response['answer']
    if 'images' in response:
        print("images are thiere...")
    else:
        print(response.keys())
    with open(convo_txt, 'a') as f:
        f.write('User: ' + ctx + '\n')
        f.write('Bot: ' + answer + '\n')

    # Build context list for frontend
    context_list = []
    if 'images' in response and response['images']:
        context_list.append({
            "text": answer,       # or clipped context text if available
            "images": response['images']
        })

    return jsonify({
        "response": response["answer"],
        "context": context_list
    })


# @agent.tool  
# def create_stl(ctx: RunContext[str]) -> str:
def create_stl(ctx: str) -> str:
    """Create an STL file given the user's specifications."""
    print("Creating an STL file...")
    # example_stls = ["/Users/blambright/Downloads/Capstone/main_code/e-NABLE_Phoenix_Hand_v3-4056253/files/Arm_Guard.stl",
    #                 "/Users/blambright/Downloads/Capstone/main_code/e-NABLE_Phoenix_Hand_v3-4056253/files/Distals.stl"]
    # response = ragbot.query(ctx, rag_chain, get_images=True, doc_store=doc_store)
    # change this to be a string response from the llm
    length_response = length_finder.run_sync(ctx).output
    width_response = width_finder.run_sync(ctx).output
    try:
        print(length_response, width_response)
        int(length_response)
        int(width_response)
    except:
        return jsonify({
            "response": "Could not determine length and width from the description provided. Please provide more specific measurements.",
            "context": []
        })
    scale_factor = calculate_scale(float(length_response), float(width_response))
    stl_scaler = MeshScaler(scale_factor)
    example_stls = stl_scaler.process_all_meshes()
    zip_id = str(uuid.uuid4())
    zip_filename = f"{zip_id}.zip"
    zip_path = os.path.join("static/files/", zip_filename)
    # with open(output_path, "w") as f:
    #     f.write("Hello! This is generated file content.")
    # shutil.copy(example_stl, output_path)
    # Build the zip
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in example_stls:
            # name inside the zip (avoid leaking full local paths)
            arcname = os.path.basename(src)
            zf.write(src, arcname=arcname)

    file_url = url_for("serve_file", filename=zip_filename, _external=True)
    return jsonify({
        "response": "STL file created successfully.",
        "file_url": file_url,
        "context": []
    })

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory("static/files", filename)

@app.route("/chat", methods=["POST"])
def get_tutor_response():
    user_msg = request.json['message']

    # Use agent to decide tool
    chat_result = agent.run_sync(user_msg).output
    print("Agent decided:", chat_result)

    if "1" in chat_result:
        return query_doc(user_msg)
    elif "2" in chat_result:
        return create_stl(user_msg)
    else:
        return jsonify({"response": general_answer.run_sync(user_msg).output, "context": []})

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

    # return jsonify({
    #     "response": response["answer"],
    #     "context": context_list
    # })




if __name__ == "__main__":
    load_credentials()
    app.run(host='0.0.0.0', port=4999, debug=False)


"""
don't forget to export Google API Key and HF_TOKEN before running
"""
import os
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

import json
from pathlib import Path
from tempfile import mkdtemp

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from langchain_google_genai import ChatGoogleGenerativeAI

import matplotlib.pyplot as plt
from PIL import ImageDraw

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint

from docling.chunking import DocMeta
from docling.datamodel.document import DoclingDocument

from io import BytesIO
import base64

class RAGBot:
    def __init__(self, source_dir):
        load_dotenv()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        HF_TOKEN = self._get_env_from_colab_or_os("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable not set")

        script_dir = Path(__file__).resolve().parent
        # print(script_dir)
        # print("CWD:", os.getcwd())
        # print(source_dir)
        self.sources = os.listdir(os.path.join(script_dir, source_dir))
        self.sources = [os.path.join(script_dir, source_dir, f) for f in self.sources]
        print(f"Sources: {self.sources}")
        self.embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        # self.gen_model_id = "Qwen/Qwen2.5-7B-Instruct"
        # PROMPT = PromptTemplate.from_template(
        #     "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
        # )
        self.top_k = 3
        self.milvus_uri = str(Path(mkdtemp()) / "docling.db")

    def _get_env_from_colab_or_os(self, key):
        return os.getenv(key)

    def process_documents(self):
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        generate_page_images=True,
                        images_scale=2.0,
                    ),
                )
            }
        )

        doc_store = {}
        doc_store_root = Path(mkdtemp())
        for source in self.sources:
            dl_doc = converter.convert(source=source).document
            file_path = Path(doc_store_root / f"{dl_doc.origin.binary_hash}.json")
            dl_doc.save_as_json(file_path)
            doc_store[dl_doc.origin.binary_hash] = file_path

        loader = DoclingLoader(
            file_path=self.sources,
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer=self.embed_model_id),
        )

        docs = loader.load()

        # for d in docs[:3]:
        #     print(f"- {d.page_content=}")
        # print("...")

        embedding = HuggingFaceEmbeddings(model_name=self.embed_model_id)

        milvus_uri = str(Path(mkdtemp()) / "docling.db")  # or set as needed
        vectorstore = Milvus.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name="docling_demo",
            connection_args={"uri": milvus_uri},
            index_params={"index_type": "FLAT"},
            drop_old=True,
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        return retriever, doc_store

    def clip_text(self, text, threshold=100):
        return f"{text[:threshold]}..." if len(text) > threshold else text

    def build_bot(self, retriever: Milvus, prompt: PromptTemplate):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    def query(self, input_query: str, rag_chain, get_images: bool = False, doc_store=None):
        resp_dict = rag_chain.invoke({"input": input_query})

        clipped_answer = self.clip_text(resp_dict["answer"], threshold=500)
        text = f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}"
        # print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}")

        if get_images:
            return self.get_images(resp_dict, doc_store, text)

        return {"answer": text}

    def get_images(self, resp_dict, doc_store, text: str = None):
        result = {}

        for i, doc in enumerate(resp_dict["context"][:]):
            image_by_page = {}
            print(f"Source {i + 1}:")
            print(f"  text: {json.dumps(self.clip_text(doc.page_content, threshold=350))}")
            meta = DocMeta.model_validate(doc.metadata["dl_meta"])

            # loading the full DoclingDocument from the document store:
            dl_doc = DoclingDocument.load_from_json(doc_store.get(meta.origin.binary_hash))

            for doc_item in meta.doc_items:
                if doc_item.prov:
                    prov = doc_item.prov[0]  # here we only consider the first provenence item
                    page_no = prov.page_no
                    if img := image_by_page.get(page_no):
                        pass
                    else:
                        page = dl_doc.pages[prov.page_no]
                        print(f"  page: {prov.page_no}")
                        img = page.image.pil_image
                        image_by_page[page_no] = img
                    bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
                    bbox = bbox.normalized(page.size)
                    thickness = 2
                    padding = thickness + 2
                    bbox.l = round(bbox.l * img.width - padding)
                    bbox.r = round(bbox.r * img.width + padding)
                    bbox.t = round(bbox.t * img.height - padding)
                    bbox.b = round(bbox.b * img.height + padding)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(
                        xy=bbox.as_tuple(),
                        outline="blue",
                        width=thickness,
                    )
            # for p in image_by_page:
            #     img = image_by_page[p]
            #     plt.figure(figsize=[15, 15])
            #     plt.imshow(img)
            #     plt.axis("off")
            #     plt.show()

            # convert images to base64
            images_base64 = [self.pil_to_base64(img) for img in image_by_page.values()]

            result["source_index"] = i
            result["answer"] = text
            result["images"] = images_base64

        return result
    
    def pil_to_base64(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # or JPEG
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

def get_prompt_template():
    prompt = PromptTemplate.from_template(
        "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
    )
    return prompt

def main():
    ragbot = RAGBot(source_dir="pages")
    retriever, doc_store = ragbot.process_documents()
    prompt = get_prompt_template()
    rag_chain = ragbot.build_bot(retriever, prompt)
    ragbot.query("What are the steps to assemble the Phoenix prosthetic arm?", rag_chain, get_images=False, doc_store=doc_store)
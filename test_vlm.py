import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
image_path, question = sys.argv[1], sys.argv[2]
mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

vlm = ChatGoogleGenerativeAI(
    model=os.getenv("VLM_MODEL", "gemini-3.5-flash-lite"),
    temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"), max_retries=3,
)
message = HumanMessage(content=[
    {"type": "text", "text": question},
    {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
])
print(vlm.invoke([message]).content)
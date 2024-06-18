# This is script for ollama.ai
# this will be used for ollama.ai api
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
import os

app = Flask(__name__)

os.system("curl https://ollama.ai/install.sh | sh")

# https://github.com/jmorganca/ollama/issues/1997#issuecomment-1892948729
#!curl https://ollama.ai/install.sh | sed 's#https://ollama.ai/download#https://github.com/jmorganca/ollama/releases/download/v0.1.20#' | sh
os.system("sudo apt install -y neofetch")

os.system("neofetch")

# you can change the model you want
OLLAMA_MODEL='phi3'

# Set it at the OS level
import os
os.environ['OLLAMA_MODEL'] = OLLAMA_MODEL
os.system("echo $OLLAMA_MODEL")

import subprocess
import time

# Start ollama as a backrgound process
command = "nohup ollama serve&"

# Use subprocess.Popen to start the process in the background
process = subprocess.Popen(command,
                            shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
print("Process ID:", process.pid)
# Let's use fly.io resources
#!OLLAMA_HOST=https://ollama-demo.fly.dev:443
time.sleep(5)  # Makes Python wait for 5 seconds

os.system("pip install langchain_community ollama langchain")

# downloading the model
os.system("ollama pull phi3")

llm = Ollama(model="phi3")

@app.route('/', methods=['POST'])
def hit_llm():
    # Get the JSON data from the request
    data = request.get_json()

    # Check if 'username' key exists in the JSON data
    if 'text_corpus' in data:
        curr_text = data['text_corpus']
        message = llm.invoke(curr_text)
        return jsonify(message)
    else:
        return jsonify({'error': 'Invalid data'}), 400

if __name__ == '__main__':
    app.run(debug=True)

# to run this script use following command
# python api.py
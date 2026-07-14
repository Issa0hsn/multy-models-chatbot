# multy-models-chatbot
A smart chatbot based on local models to reduce dependence on external servers.
## this chatbot use:
  * Qwen open-weights model working on your own hardware (or hf space) what gives you the independence and full control but a little bet slower.
  * GPT-5 as proprietary model (closed-weights) working on OpenAI servers that makes it leighter and faster but you have no more access to the weights like open-weights .

# Learning Journal
___
## 1-AI principles: 
### LLMs: I learned about language models and how it generate the text ,it turns out that it predicts the next word baswd on its probability and on some paramters like "temperature","top_p","top_k".
  *temperature: it controls the randomness of the choice (increasing it make the chat bot more creative)
  *top_p: increasing it let gives the model words to choose between as the next word.
  *for short we can say that these parameters forms the chatbot personality.
### Model Size & Precision : the power of the model generaly dependence on its size and how it is loaded (16-bit,INT8....)
### Context Length:
  * models don't understands the human words instead it convert it to "tokens" , the more "token" it use the large amount of memory (KV Cache) it needs to handel to generate the answer, the chat  history also use  some of the context length.
  * when the context length end the model starts to forget some of the chat history or crashes if the error wasn't handeled correctly.
### System vs User Prompt:
  * user prompt : a direct request for information the model answar temporarily.
  * system prompt : defining the identity and behavior of the model. 
## 2-Ecosystem:
### Hugging Face: the most important tool for any ML project it helps me comparing the models hosting the project and watching other peopel progect and inspired by.
### Gradio: powerful and simple tool to make interfaces, using features like streaming just by adding some prameters is so cool.

## 3-infrence & tuning:
### Thinking Model: models do not just predicte the next word ,it also can make the "Reasoning" process to break down the answer into small logical steps to improve the performance and  reduce the hallucinations rate. 
### inference (cpu vs gpu): there is massive difference in speed between cpu and gpu
  * hallucination : the model gives you a wrong informations or incomprehensible sentences confidently(thats because it predicte the most linguistically probable words not logiclly).  
# Technical Challenges and Solutions:
## 1-OOM(Out Of Memory):
it means when the available resources end or the conversation exceed the available context length.
big models on a insufficient memory(RAM/VRAM) would be allways cause this problem so I choise a suitable model fore memory size and cpu power.

# how to run:
___
git clone https://github.com/Issa0hsn/multy-models-chatbot 

cd chatbot

pip install -r requirements.txt

python chatbot2.py

or just [ visit my space](https://huggingface.co/spaces/isahsn/MultyModelsChatbot)

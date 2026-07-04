## multy-models-chatbot
A smart chatbot based on local models to reduce dependence on external servers.
this chatbot use qwen 3.5 -2b as a local model and gpt5 by api.

## Learning Journal
______________________________________________________________________________________________________________________________________________________________________
# 1-AI principles: 
LLMs: I learned about language models and how it generate the text ,it turns out that it predicts the next word baswd on its posiblity and on some prameters like "temperature" "top_p" "top_k".
Model Size : the power of the model generaly dependence on its size and how it is loaded (16 byte ,32byte....)
Context Length: models don't understands the human words instade it convert it to "tokens" , the more "token" it use the more it working to generate the answer.

# 2-Ecosystem:
Hugging Face: the most important tool for any ML progect it helps me comparing the models hosting the project and watching other peopel progect and inspired by.
Gradio: powerful and simple tool to make interfaces, using features like streming just by adding some prameters is so cool.

# 3-infrence & tuning:
Thinking Model:
inference (cpu vs gpu): ther is masive difference in speed between cpu and gpu

## Technical Challenges and Solutions:
# 1-OOM:
big models on a small spaces would be allways cause this problem so I choise a suitable model fore memory size and cpu power.

## how to run:
__________________________________________________________________________________________________________________________________________________________________
git clone ....
pip install -r requirements.txt

or go to https://huggingface.co/spaces/isahsn/MultyModelsChatcot

# Health_Care_Assistant_ChatBot-End-to-End-RAG-Implementaion-
A Medical Assistant chatbot powered by Retrieval-Augmented Generation (RAG).


## 🚀 How to Run  

Follow these steps to set up and run the project locally:

---

### 1️. Clone the Repository  

```bash
# Clone this repository
git clone https://github.com/shreejoysarkar/Health_Care_Assistant_ChatBot-End-to-End-RAG-Implementaion.git
```

### 2. Create a uv enrionment after opening the repository
```bash
## initializing uv 
## (paste these command in the cmd)

uv init

uv venv medibot --python 3.13

#### Activating the env

medibot\Scripts\activate
```

### 3. Installing requirements

```bash
uv pip install -r requirements.txt
```


This project uses medical reference data from Davidson’s Principles & Practice of Medicine AND Gale Encyclopedia of Medicine.

The PDF is not included in this repository due to GitHub’s file size limitations.

Download it from here:
https://drive.google.com/drive/folders/1ogzqlXf0vpLxMO4oZCT0dxlQbRcKdD2O?usp=sharing


After downloading, place the file inside a Data/ folder in the root directory of the project.
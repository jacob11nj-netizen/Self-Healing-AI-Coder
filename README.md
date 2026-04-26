#  Self-Healing AI Engineer

A professional AI agent built with **LangGraph** and **Streamlit** that doesn't just write code—it **validates and fixes it** autonomously.

###  Why this is special
Most AI tools just generate text. If the code is wrong, it stays wrong. This agent:
1.  **Writes** Python code based on a prompt.
2.  **Tests** that code in a real Python environment.
3.  **Analyzes** the error if the code crashes.
4.  **Heals** itself by rewriting the code until it works.

###  Tech Stack
* **Brain:** Llama 3.1 (via Groq API)
* **Logic:** LangGraph (State management and loops)
* **Interface:** Streamlit (Web Dashboard)
* **Language:** Python 3.14

###  How to Run Locally
1. Clone the repo.
2. Add your `GROQ_API_KEY` to a `.env` file.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
# E-Commerce_Customer_Agent

## Overview
The E-Commerce Customer Agent is a tool designed to enhance customer support for e-commerce platforms. It provides automated assistance to customers, handling common inquiries such as order tracking, product information, returns, and FAQs. The agent aims to improve user experience by offering quick and accurate responses, reducing the workload on human support teams.

```
+----------------------------+        +----------------------+        +-------------------+
|    Client (Web/Mobile)     | <----> |  API / Gateway (LB)  | <----> |  Orchestrator     |
|  - Chat UI / Voice / SMS   |        |  - Auth, Rate limit  |        |  - Intent Router   |
+----------------------------+        +----------------------+        +---------+---------+
                                                                       |         |
                                                                       |         |
                           +-------------------------------+-----------+         +--------------------------+
                           |                               |                             |
                 +---------v---------+           +---------v--------+           +--------v---------+
                 | Cancelling Agent  |           | Tracking Agent   |           | ProductInfo Agent|
                 | - Domain logic    |           | - Order queries   |           | - RAG: retriever |
                 | - Auth checks     |           | - Shipping APIs   |           |   + LLM (fusion)  |
                 +---------+---------+           +---------+--------+           +--------+---------+
                           |                               |                             |
                           |                               |                             |
                    +------v------+                 +------v------+               +------v------+
                    | Order DB /  |                 | Shipping/    |               | VectorStore  |
                    | OMS        |                 | Carrier APIs  |               | + Knowledge  |
                    +-------------+                 +-------------+               +-------------+

```

1. Orchestrator Agent: This is used to detect intent, routes to the agents as shown in the figure based on the intent. It is also used to maintain conversational context and memory during multi-turns. 
2. Cancelling Agent: This helps in cancelling orders by interacting with a mock API
3. Tracking Agent: This gives the status of the orders by interacting with a mock API
4. ProductInfo Agent: This is a RAG system which helps with general FAQ. It is based on the FAISS vector database and uses semantic and lexical search

Multi-Turn
```
User -> API: POST /message (session_id)
API -> Orchestrator: validated message
Orchestrator -> Redis: GET session
Orchestrator -> LLM: call(prompt with context)
LLM -> Orchestrator: structured response
Orchestrator -> PolicyEngine: validate(response)
Orchestrator -> ActionRunner(s): execute actions
ActionRunner -> Orchestrator: results
Orchestrator -> Redis: update session history + pending_action/results
Orchestrator -> API -> User: final reply
```


## Installation (use python3.10 and above)

Without using Docker,


From terminal, clone the repository: 

```bash
git clone https://github.com/banriskhem-kk/BKK-CustomerAgent.git
```

Navigate to the project directory:cd BKK-CustomerAgent

Install dependencies
```
Make sure you have Ollama installed https://ollama.com/download

ollama pull <model_name>
```

```
pip install -r requirements.txt
```
Configure environment variables (e.g., API keys, database connections) in a .env file.

```
BEECEPTOR_URL="https://banri.proxy.beeceptor.com/"
REDIS_PASSWORD="xxxx"
REDIS_HOST=""
REDIS_PORT=1234
REDIS_USERNAME=default
```

Usage

Start the application in one terminal:
```bash
python run.py
```
Run Command for user request in another terminal:

```bash
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id": "<Session ID>", "message": "<Your Message>"}'
```

Post orders to Mock API
```bash
curl -X POST https://banri.proxy.beeceptor.com/orders/ORD-2400 -H "Content-Type: application/json" -d '{
  "order_id": "ORD-2400",
  "status": "paid",
  "product": "Headphones",
  "product_id": "BT001",
  "product_name": "Bluetooth Headphones",
  "product_quantity": 2, "cancel_time": 24, "block_hours":4, "refund_flag": true
}'
```

## Interaction


### User:

```json
{
  "session_id": "axb189",
  "message": "Can I return my Bluetooth headphones?"
},
```

### Agent response:

```json
{
  "response": "Yes, you can return your Bluetooth headphones. According to our return policy, you can return them within 30 days of purchase with a valid receipt. Please note that the items must be in their original condition and packaging for a full refund.\n\nTo initiate the return process, I recommend checking out our Returns page on our website for detailed instructions. This will guide you through the steps to ensure your return is processed smoothly.\n\nIf you have any questions or concerns about the return process, feel free to ask!",
  "agent": "ProductInfoAgent",
  "tool_calls": [],
  "handover": "ProductInfoAgent → OrchestratorAgent"
},
```

### User:

```json
{
  "session_id": "axb189",
  "message": "Alright, cancel the order for that then."
},
```

### Agent response:

```json
{
  "response": "Please provide an order ID (e.g., ORD-1234) to cancel your order.",
  "agent": "OrchestratorAgent",
  "tool_calls": [],
  "handover": "OrchestratorAgent"
},
```

### User:

```json
{ 
  "session_id": "axb189", 
  "message": "The order id is ORD-2500" }
```

### Agent response:

```json
{
  "response": "Your order ORD-2500 has been successfully canceled.",
  "agent": "OrderCancellationAgent",
  "tool_calls": [
    {
      "tool": "OrderCancellationAPI",
      "input": { "orderId": "ORD-2500" },
      "result": {
        "status": "cancelled",
        "refunded": true,
        "order_id": "ORD-2500"
      }
    }
  ],
  "handover": "OrderCancellationAgent → OrchestratorAgent"
}
```

Modularity of LLM:
```
LLM can be interchangeable to Cloud API Based models like OpenAI or OpenRounter, or more. 

All changes to LLM response can be done in the llm.py file, 

Changes are to be make in the def chat function. If you want to implement a different LLM response calling (e.g Using OpenAI, or others)
If using Cloud Based APIs, store APIs in .env file and call using os.getenv from the llm.py
```
you can then add the following to your .env file:

```
OPENAI_API=""
OPENROUTER_API=""
```

Docker based (sub-optimal),

Two files are needed, 

```
dockerfile
docker-compose.yaml
```

Then run:

```
docker-compose up --build
```
Run Command for user request in another terminal

```
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id": "abc12345", "message": "Can you tell me about my order"}'
```








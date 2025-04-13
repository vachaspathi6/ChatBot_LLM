import streamlit as st
import time
import boto3
from openai import OpenAI, AzureOpenAI
from google import genai
import json
import os
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, ToxicityMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

os.environ["OPENAI_API_KEY"] = "Place Your OpenAI API KEY"

def get_model_response(customer_input, aws_access_key_id, aws_secret_access_key, boto_session):
    if not aws_access_key_id and not aws_secret_access_key:
        st.info("🔑 Access Key Id or Secret Access Key are not provided yet!")
        return None

    client = boto_session.client(
        service_name='bedrock-runtime',
        region_name="us-east-1"
    )

    prompt = f"\n\nHuman:{customer_input}\n\nAssistant:"

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1000,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
    })

    response = client.invoke_model(
        body=body,
        modelId="amazon.titan-text-premier-v1:0",
        accept='application/json',
        contentType='application/json'
    )

    msg = json.loads(response['body'].read().decode('utf-8'))
    response_text = msg['results'][0]['outputText']

    return response_text

# Function to load OpenAI model and get response
def get_chatgpt_response(api_key, question):
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o",
        input=question
    )
    return response.output_text

# Function to load Gemini model and get response
def get_gemini_response(api_key, question):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=question
    )
    return response.text

# Function to load Azure OpenAI GPT-3.5 Turbo model and get response
def get_azure_gpt_response(api_base, api_version, api_key, question):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base,
    )

    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "How can I help you?"},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9
    )

    return response.choices[0].message.content

## Initialize our Streamlit app
st.set_page_config(
    page_title="LLM Chatbot Models",
    page_icon="🤖",
    layout="centered"
)
st.title("💬 Research on LLM Models")
st.caption("🚀 A dynamic conversational experience powered by various LLMs.")

# Sidebar for model selection and API key input
st.sidebar.header("🔧 Configuration")
model_choice = st.sidebar.radio("🎛️ Choose the model:", ["Gemini Pro", "GPT-4.0", "Azure OpenAI GPT-3.5 Turbo","AWS"])
api_key = None

if model_choice == "Gemini Pro":
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
elif model_choice == "GPT-4.0":
    api_key = st.sidebar.text_input("🔑 Chat-GPT API Key", type="password")
elif model_choice == "AWS":
    aws_access_key_id = st.sidebar.text_input("🔑 AWS Access Key Id", placeholder="access key", type="password")
    api_key = st.sidebar.text_input("🔑 AWS Secret Access Key", placeholder="secret", type="password")   
else:
    api_base = st.sidebar.text_input("🔑 Amazon API Base URL", placeholder="https://<name>.openai.azure.com/")
    api_version = st.sidebar.text_input("🔑 API Version", "2023-03-15-preview")
    api_key = st.sidebar.text_input("🔑 API Key", type="password")

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for message in st.session_state.chat_history:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

# Clear history button and input field for user's message side by side
col1, col2 = st.columns([7, 3])
if col2.button("Clear Chat History"):
    st.session_state.chat_history = []

input_text = col1.chat_input("Ask your question here:")

if input_text:
    if api_key:
        # Add user's message to chat and display it
        st.session_state.chat_history.append({"role": "user", "content": input_text})
        st.chat_message("user").markdown(input_text)

        # Send user's message to the selected AI model and get a response
        start_time = time.time()
        if model_choice == "Gemini Pro":
            response = get_gemini_response(api_key, input_text)
        elif model_choice == "GPT-4.0":
            response = get_chatgpt_response(api_key, input_text)
        elif model_choice == "Azure OpenAI GPT-3.5 Turbo":
            response = get_azure_gpt_response(api_base, api_version, api_key, input_text)
        elif model_choice=="AWS":
            boto_session = boto3.session.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=api_key)

            response=get_model_response(input_text,aws_access_key_id,api_key,boto_session)

        end_time = time.time()

        assistant_response = response
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Calculate evaluation metrics
        latency = end_time - start_time
        input_tokens = len(input_text.split())
        output_tokens = len(assistant_response.split())
        throughput = output_tokens / latency

        test_case = LLMTestCase(input=input_text, actual_output=response)
        relevancy_score = evaluate(test_cases=[test_case], metrics=[AnswerRelevancyMetric(threshold=0.7)])
        bias_score = evaluate(test_cases=[test_case], metrics=[BiasMetric(threshold=0.5)])
        toxicity_score = evaluate(test_cases=[test_case], metrics=[ToxicityMetric(threshold=0.5)])
        correctness_score = evaluate(test_cases=[LLMTestCase(input=input_text, actual_output=response, tools_called=[ToolCall(name="WebSearch"), ToolCall(name="ToolQuery")], expected_tools=[ToolCall(name="WebSearch")],)], metrics=[ToolCorrectnessMetric(threshold=0.5)])

        # Display the AI's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Display metrics in the sidebar
        st.sidebar.subheader("📊 Evaluation Metrics")
        st.sidebar.write(f"- **Throughput:** {throughput:.6f} tokens/second")
        st.sidebar.write(f"- **Latency:** {latency:.6f} seconds")
        st.sidebar.write(f"- **Input Tokens:** {input_tokens}")
        st.sidebar.write(f"- **Output Tokens:** {output_tokens}")
        st.sidebar.write(f"- **Relevancy Score:** {relevancy_score.test_results[0].metrics_data[0].score}")
        st.sidebar.write(f"- **Bias Score:** {bias_score.test_results[0].metrics_data[0].score}")
        st.sidebar.write(f"- **Toxicity Score:** {toxicity_score.test_results[0].metrics_data[0].score}")
        st.sidebar.write(f"- **Correctness Score:** {correctness_score.test_results[0].metrics_data[0].score}")

    else:
        st.sidebar.error("⚠️ Please enter your API key to proceed.")

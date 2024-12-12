import traceback
import uuid
import sys
from typing import TypedDict, Sequence, List

from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain.prompts import ChatPromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langgraph.checkpoint.memory import MemorySaver
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from models import get_model_id, BEDROCK_PRICING, US_TO_JPY_RATE


class LLMConfig:
    MODEL_ID: str
    MODEL_NAME: str = "claude-v3.5-sonnet-v2"
    MAX_TOKENS: int = 5120
    SYSTEM_PROMPT: str = "You are a helpful assistant. Answer all questions to the best of your ability."
    REGION: str = "us-west-2"

LLMConfig.MODEL_ID = get_model_id(LLMConfig.MODEL_NAME)

class ChatState(TypedDict):
    messages: Sequence[BaseMessage]
    thread_id: str
    total_tokens: int
    total_cost: float

class ChatBot:
    def __init__(self):
        self.llm = self._create_llm()
        self.rules = self._get_rules()
        self.prompt = self._create_prompt()
        self.graph = self._create_workflow()
        self.thread_id = str(uuid.uuid4())
        self.total_tokens = 0
        self.total_cost = 0.0
        self.state = None

    def _create_llm(self):
        return ChatBedrock(
            model_id=LLMConfig.MODEL_ID,
            model_kwargs={
                "max_tokens": LLMConfig.MAX_TOKENS,
                "system": LLMConfig.SYSTEM_PROMPT
            },
            region_name=LLMConfig.REGION
        )

    def _get_rules(self):
        return "- don't explain details"

    def _create_prompt(self):
        return ChatPromptTemplate.from_template("""
        Previous conversation:
        {messages}

        Current input:
        {input}

        Rules:
        {rules}

        Please provide a response following the rules above.
        """)

    def _create_workflow(self):
        workflow = StateGraph(ChatState)
        workflow.add_node("process", self._process_message)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        workflow.set_entry_point("process")
        return workflow.compile(checkpointer=MemorySaver())

    def _trim_conversation_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        return trim_messages(
            messages,
            max_tokens=4096,
            strategy="last",
            token_counter=self.llm.get_num_tokens_from_messages,
            include_system=True,
            allow_partial=False,
            start_on="human"
        )

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        input_price = (
            BEDROCK_PRICING.get(LLMConfig.REGION, {})
            .get(model_name, {})
            .get("input", BEDROCK_PRICING["default"][model_name]["input"])
        )
        output_price = (
            BEDROCK_PRICING.get(LLMConfig.REGION, {})
            .get(model_name, {})
            .get("output", BEDROCK_PRICING["default"][model_name]["output"])
        )
        input_cost = (input_tokens / 1000) * input_price
        output_cost = (output_tokens / 1000) * output_price
        return ( input_cost + output_cost )* US_TO_JPY_RATE

    def _process_message(self, state: ChatState, config: dict) -> ChatState:
        messages = list(state["messages"])
        messages.append(HumanMessage(content=config["configurable"]["input"]))

        messages = self._trim_conversation_history(messages)
        formatted_messages = self._format_messages(messages)

        prompt_messages = self.prompt.format_messages(
            messages=formatted_messages,
            input=config["configurable"]["input"],
            rules=self.rules
        )

        input_tokens = self.llm.get_num_tokens_from_messages(prompt_messages)
        response = self.llm.invoke(prompt_messages)
        output_tokens = self.llm.get_num_tokens_from_messages([AIMessage(content=response.content)])

        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens, LLMConfig.MODEL_NAME)

        self.total_tokens += total_tokens
        self.total_cost += cost


        messages.append(AIMessage(content=response.content))

        return {
            "messages": messages,
            "thread_id": state["thread_id"],
            "total_tokens": total_tokens,
            "total_cost": cost
        }

    def chat(self, source_text: str) -> tuple[str, int, float]:
        if not source_text.strip():
            raise ValueError("Input text cannot be empty")

        if not self.state:
            self.state = {
                "messages": [],
                "thread_id": self.thread_id,
                "total_tokens": 0,
                "total_cost": 0.0
            }

        config = {
            "configurable": {
                "input": source_text,
                "thread_id": self.thread_id
            }
        }

        result = self.graph.invoke(self.state, config)
        self.state = result

        return (
            result["messages"][-1].content,
            result["total_tokens"],
            result["total_cost"]
        )

class ChatInterface:
    def __init__(self):
        self.console = Console()
        self.chatbot = ChatBot()
        self.prompt_session = PromptSession[str]()

    def _create_key_bindings(self):
        bindings = KeyBindings()

        def exit_handler(event: KeyPressEvent):
            print("\nGoodbye!")
            sys.exit(0)

        bindings.add("c-c")(exit_handler)
        bindings.add("c-d")(exit_handler)

        return bindings

    def prompt(self):
        return self.prompt_session.prompt(
            "You> ",
            vi_mode=False,
            multiline=True,
            enable_open_in_editor=True,
            key_bindings=self._create_key_bindings(),
            prompt_continuation=lambda width, line_number, is_soft_wrap: '... '
        )

    def run(self):
        self.console.print("Chat started. Enter your message (press Ctrl+D to send, Ctrl+C to save and exit):")

        while True:
            user_input = self.prompt()
            if not user_input.strip():
                continue

            try:
                answer, tokens, cost = self.chatbot.chat(user_input)

                self.console.print(f"Bedrock: {answer}")
                self.console.print(Panel(Text(
                    f"Tokens used: {tokens:,}\n"
                    f"Cost: {cost:.4f}\n"
                    f"Total cost: {self.chatbot.total_cost:.4f}",
                    style="bold green"
                )))

            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
                self.console.print("[red]Stacktrace:[/red]")
                self.console.print(traceback.format_exc())

def main():
    interface = ChatInterface()
    interface.run()

if __name__ == "__main__":
    main()

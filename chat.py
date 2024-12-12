import traceback
import uuid
import sys
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain.prompts import ChatPromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

class ChatState(TypedDict):
    messages: Sequence[BaseMessage]
    thread_id: str

class ChatBot:
    def __init__(self):
        self._init_llm()
        self._init_rules()
        self._init_prompt()
        self._setup_workflow()
        self.thread_id = str(uuid.uuid4())

    def _init_llm(self):
        """Initialize the Bedrock chat model"""
        self.llm = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            #model_id="amazon.nova-pro-v1:0",
            model_kwargs={
                "max_tokens": 5120,
                "system": "You are a helpful assistant. Answer all questions to the best of your ability."
            },
            region_name="us-west-2"
            #region_name="us-east-1"
        )

    def _init_rules(self):
        """Initialize chat rules"""
        self.rules = """
        - don't explain details
        """

    def _init_prompt(self):
        """Initialize prompt template"""
        self.prompt = ChatPromptTemplate.from_template("""
        Previous conversation:
        {messages}

        Current input:
        {input}

        Rules:
        {rules}

        Please provide a response following the rules above.
        """)

    def _setup_workflow(self):
        """Set up conversation workflow graph"""
        workflow = StateGraph(ChatState)
        workflow.add_node("process", self._process_message)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        workflow.set_entry_point("process")
        self.graph = workflow.compile(checkpointer=MemorySaver())

    def _trim_conversation_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Trim conversation history to fit token limit"""
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
        """Format message history into string"""
        return "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        )

    def _process_message(self, state: ChatState, config: dict) -> ChatState:
        """Process a single message exchange"""
        messages = list(state["messages"])
        messages.append(HumanMessage(content=config["configurable"]["input"]))

        # Trim history before formatting to ensure we stay within token limits
        messages = self._trim_conversation_history(messages)
        formatted_messages = self._format_messages(messages)

        # Pass the current input separately to ensure it's not lost in history trimming
        response = self.llm.invoke(
            self.prompt.format_messages(
                messages=formatted_messages,
                input=config["configurable"]["input"],
                rules=self.rules
            )
        )

        messages.append(AIMessage(content=response.content))
        return {"messages": messages, "thread_id": state["thread_id"]}

    def chat(self, source_text: str) -> str:
        """Process user input and return AI response"""
        if not source_text.strip():
            raise ValueError("Input text cannot be empty")

        if not hasattr(self, 'state'):
            self.state = {"messages": [], "thread_id": self.thread_id}

        config = {
            "configurable": {
                "input": source_text,
                "thread_id": self.thread_id
            }
        }
        result = self.graph.invoke(self.state, config)

        # Update the state with the new messages
        self.state = result

        return result["messages"][-1].content

class ChatInterface:
    def __init__(self):
        self.console = Console()
        self.chatbot = ChatBot()
        self.prompt_session = PromptSession[str]()

    def prompt(self):
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _(event: KeyPressEvent):
            print("\nGoodbye!")
            sys.exit(0)

        @bindings.add("c-d")
        def _(event: KeyPressEvent):
            print("\nGoodbye!")
            sys.exit(0)

        return self.prompt_session.prompt(
            "You> ",
            vi_mode=False,
            multiline=True,
            enable_open_in_editor=True,
            key_bindings=bindings,
            prompt_continuation=self.prompt_continuation
        )

    def prompt_continuation(self, width, line_number, is_soft_wrap):
        return '... '

    def run(self):
        """Run the chat interface"""
        self.console.print("Chat started. Enter your message (press Ctrl+D to send, Ctrl+C to save and exit):")
        while True:
            user_input = self.prompt()
            if not user_input.strip():  # Skip completely empty inputs
                continue
            try:
                answer = self.chatbot.chat(user_input)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.console.print(f"Bedrock: {answer}")
                self.console.print(Panel(Text(f"{current_time}", style="bold green")))
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
                self.console.print("[red]Stacktrace:[/red]")
                self.console.print(traceback.format_exc())

def main():
    interface = ChatInterface()
    interface.run()

if __name__ == "__main__":
    main()

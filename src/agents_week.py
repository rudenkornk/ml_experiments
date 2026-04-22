# ruff: noqa: D100, D101, D102, D417, D401, D107, T201, ANN204, ANN401, ANN001, D205, D212, PLR0913, ERA001, E402
# ╔══════════════════════════════════════════════════════════════╗
# ║          TEMPLATE — DO NOT MODIFY THIS CELL                 ║
# ╚══════════════════════════════════════════════════════════════╝

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-oss-20b"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE")
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


def llm_chat(messages: list, tools: list | None = None) -> AIMessage:
    """Sends the message history to the LLM and returns the model response.

    Parameters
    ----------
      messages — list of dialog messages. Each message is a LangChain object:
                   SystemMessage(content="...")   — instruction for the model (agent role)
                   HumanMessage(content="...")    — message from the user
                   AIMessage(...)                 — previous model response
                   ToolMessage(content="...", tool_call_id="...") — tool result

      tools   — list of tool descriptions (OpenAI function calling schema or LangChain tools).

    Returns AIMessage:
      msg.content    — text response (str)
      msg.tool_calls — list of tool calls:
                         "name" — tool name
                         "args" — arguments (already parsed dict)
                         "id"   — unique call identifier

    """
    if tools:
        return llm.bind_tools(tools).invoke(messages)
    return llm.invoke(messages)


# Product catalog
CATALOG = [
    {
        "id": "p1",
        "name": "Sony WH-1000XM5",
        "category": "headphones",
        "brand": "Sony",
        "price": 349,
        "color": "black",
        "rating": 4.8,
        "tags": ["wireless", "noise-cancelling", "premium"],
    },
    {
        "id": "p2",
        "name": "Sony WH-CH720N",
        "category": "headphones",
        "brand": "Sony",
        "price": 129,
        "color": "blue",
        "rating": 4.4,
        "tags": ["wireless", "budget", "noise-cancelling"],
    },
    {
        "id": "p3",
        "name": "Bose QuietComfort Ultra",
        "category": "headphones",
        "brand": "Bose",
        "price": 379,
        "color": "white",
        "rating": 4.7,
        "tags": ["wireless", "noise-cancelling", "premium"],
    },
    {
        "id": "p4",
        "name": "Apple AirPods Pro 2",
        "category": "earbuds",
        "brand": "Apple",
        "price": 249,
        "color": "white",
        "rating": 4.6,
        "tags": ["wireless", "noise-cancelling", "ios"],
    },
    {
        "id": "p5",
        "name": "Anker Soundcore Liberty 4 NC",
        "category": "earbuds",
        "brand": "Anker",
        "price": 99,
        "color": "black",
        "rating": 4.3,
        "tags": ["wireless", "budget", "noise-cancelling"],
    },
    {
        "id": "p6",
        "name": "Logitech MX Master 3S",
        "category": "mouse",
        "brand": "Logitech",
        "price": 109,
        "color": "graphite",
        "rating": 4.8,
        "tags": ["wireless", "productivity", "premium"],
    },
    {
        "id": "p7",
        "name": "Logitech Pebble 2",
        "category": "mouse",
        "brand": "Logitech",
        "price": 34,
        "color": "white",
        "rating": 4.2,
        "tags": ["wireless", "budget", "portable"],
    },
    {
        "id": "p8",
        "name": "Keychron K2",
        "category": "keyboard",
        "brand": "Keychron",
        "price": 89,
        "color": "black",
        "rating": 4.5,
        "tags": ["wireless", "mechanical", "compact"],
    },
    {
        "id": "p9",
        "name": "NuPhy Air75",
        "category": "keyboard",
        "brand": "NuPhy",
        "price": 139,
        "color": "gray",
        "rating": 4.6,
        "tags": ["wireless", "mechanical", "low-profile"],
    },
    {
        "id": "p10",
        "name": "Amazon Kindle Paperwhite",
        "category": "ereader",
        "brand": "Amazon",
        "price": 149,
        "color": "black",
        "rating": 4.7,
        "tags": ["reading", "portable", "gift"],
    },
]


@dataclass
class ShopState:
    """Session state: cart and last search results."""

    cart: list = field(default_factory=list)
    last_results: list = field(default_factory=list)


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: Any = None


class ToolTracer:
    """Collects all tool calls."""

    def __init__(self):
        self.calls: list[ToolCallRecord] = []

    def record(self, name: str, args: dict, result: Any = None) -> None:
        self.calls.append(ToolCallRecord(name=name, args=args, result=result))

    def called(self, name: str) -> bool:
        return any(c.name == name for c in self.calls)

    def get_calls(self, name: str) -> list:
        return [c for c in self.calls if c.name == name]

    def print_trace(self) -> None:
        print("=== Tool Call Trace ===")
        for i, c in enumerate(self.calls, 1):
            print(f"  {i}. {c.name}({json.dumps(c.args, ensure_ascii=False)[:80]})")
            if c.result is not None:
                print(f"     -> {json.dumps(c.result, ensure_ascii=False)[:100]}")
        print("=====================")


class ShopTools:
    """Shop logic — search and add to cart."""

    def __init__(self, catalog):
        self.catalog = catalog

    def search_products(
        self,
        query: str = "",
        category: str | None = None,
        brand: str | None = None,
        max_price: float | None = None,
        sort_by: str | None = None,
    ) -> list:
        results = []
        q_words = query.lower().split() if query else []
        for item in self.catalog:
            hay = f"{item['name']} {item['category']} {item['brand']} {' '.join(item['tags'])}".lower()
            if q_words and not all(w in hay for w in q_words):
                continue
            if category and item["category"] != category:
                continue
            if brand and item["brand"].lower() != brand.lower():
                continue
            if max_price is not None and item["price"] > float(max_price):
                continue
            results.append(copy.deepcopy(item))
        if sort_by == "price_asc":
            results.sort(key=lambda x: x["price"])
        elif sort_by == "rating_desc":
            results.sort(key=lambda x: -x["rating"])
        return results

    def add_to_cart(self, state: ShopState, product_id: str, quantity: int = 1) -> dict:
        product = next((p for p in self.catalog if p["id"] == product_id), None)
        if not product:
            return {"ok": False, "error": f"Product {product_id} not found"}
        existing = next((r for r in state.cart if r["product_id"] == product_id), None)
        if existing:
            existing["quantity"] += quantity
        else:
            state.cart.append(
                {"product_id": product_id, "name": product["name"], "price": product["price"], "quantity": quantity}
            )
        return {"ok": True, "cart_size": len(state.cart)}


@dataclass
class AgentContext:
    """Shared context passed between agents in Task 3."""

    query: str
    max_price: float | None = None
    candidates: list[dict] = field(default_factory=list)
    pros: dict[str, str] = field(default_factory=dict)  # product_id -> pros description
    cons: dict[str, str] = field(default_factory=dict)  # product_id -> cons description
    best: dict | None = None
    cart_result: dict | None = None


TOOLS = ShopTools(CATALOG)
print("Template loaded.")
print(f"  Model: {MODEL_NAME}")
print(f"  Catalog: {len(CATALOG)} products")
print("  Utilities: AgentContext, ToolTracer, ShopTools, convert_to_openai_tool")
print("  LangChain: HumanMessage, SystemMessage, AIMessage, ToolMessage")

# ╔══════════════════════════════════════════════════════════════╗
# ║               YOUR CODE — THREE TASKS                        ║
# ╚══════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1. Tool-Calling Agent (ReAct loop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1.1. Define SHOP_TOOLS_SCHEMA — tool descriptions for the LLM.
#
# Below are stub functions with signatures but no descriptions.
# The LLM needs to understand what each tool does and what its parameters mean.
#
# Task: add a docstring (description + Args) to each function.
# The convert_to_openai_tool() function from the template will generate the JSON schema automatically.
# For docstring format details, see Google-style docstrings.

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


def search_products(
    query: str = "",
    category: str | None = None,
    brand: str | None = None,
    max_price: float | None = None,
    sort_by: str | None = None,
) -> list:
    """Search shop products by query, category, brand, and max price. Optionally sort results.

    Args:
        query: query to search. Usually consists of words from product name, category, brand and tags.
        category: exact category to filter by (e.g. "headphones", "mouse").
        brand: exact brand to filter by (e.g. "Sony", "Logitech").
        max_price: maximum price to filter by (e.g. 100).
        sort_by: sorting of results. Can be "price_asc" or "rating_desc". If not set, results are in catalog order.

    Returns:
        list: list of matching products.
              Each product is a dict with keys: id, name, category, brand, price, color, rating, tags.

    """


def add_to_cart(product_id: str, quantity: int = 1) -> dict:
    """Add a product to the shopping cart.

    Args:
        product_id: unique identifier of the product to add (e.g. "prod_001").
        quantity: number of units to add. Defaults to 1.

    Returns:
        dict: result of the operation with keys:
              - ok (bool): True if the product was added successfully, False otherwise.
              - cart_size (int): total number of distinct items in the cart (on success).
              - error (str): error message describing why the operation failed (on failure).

    """


# YOUR CODE HERE: generate the schema
SHOP_TOOLS_SCHEMA = [
    convert_to_openai_tool(search_products),
    convert_to_openai_tool(add_to_cart),
]


# 1.2. Implement run_shopping_agent — a ReAct shop agent.
def run_shopping_agent(user_message: str, state: ShopState, tools: ShopTools, tracer: ToolTracer) -> str:
    """
    ReAct shop agent. Receives a user message and iteratively:
      1. Calls the LLM with the history and tool schema.
      2. If the LLM returns tool_calls — executes each tool:
           search_products -> saves result to state.last_results, records in tracer
           add_to_cart     -> adds product to state.cart, records in tracer
         Adds a ToolMessage with the result to the history and repeats the loop.
      3. If tool_calls is empty — returns the text response from the LLM.
    """
    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant. "
            "Use the available tools to search for products and add them to the cart based on the user's request."
        ),
        HumanMessage(content=user_message),
    ]

    while True:
        response = llm_chat(messages, tools=SHOP_TOOLS_SCHEMA)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            if name == "search_products":
                result = tools.search_products(**args)
                state.last_results = result
            elif name == "add_to_cart":
                result = tools.add_to_cart(state, **args)
            else:
                result = {"error": f"Unknown tool: {name}"}

            tracer.record(name=name, args=args, result=result)
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call_id))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2. Memory Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROFILE_PATH = Path("user_profile.json")
# Recommended profile fields:
#   name       — user name
#   brand      — preferred brand
#   max_price  — maximum price
#   color      — preferred color
#   category   — category of interest


def load_profile(path: Path = PROFILE_PATH) -> dict:
    """Loads profile from JSON. Returns {} if file does not exist."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_profile(profile: dict, path: Path = PROFILE_PATH) -> None:
    """Saves the profile dict to a file as JSON."""
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))


def update_profile(
    key: Literal["name", "brand", "max_price", "color", "category"], value: str | float, path: Path = PROFILE_PATH
) -> dict:
    """Updates the profile with the given key and value. Loads the existing profile, updates it, and saves it back.

    Args:
        key: the profile field to update (e.g. "brand").
        value: new value for the field (e.g. "Sony" for brand, 100 for max_price).

    Returns:
        dict: updated profile

    """
    profile = load_profile(path)
    profile[key] = value
    save_profile(profile, path)
    return profile


SHOP_TOOLS_SCHEMA_WITH_MEMORY = [*SHOP_TOOLS_SCHEMA, convert_to_openai_tool(update_profile)]


def run_memory_agent(
    user_message: str,
    state: ShopState,
    tools: ShopTools,
    tracer: ToolTracer,
    history: list,
    profile_path: Path = PROFILE_PATH,
) -> tuple:
    """
    Memory agent. Extends run_shopping_agent with long-term and short-term memory.

    Long-term memory:
      - Loads profile from file (load_profile) on each run
      - Passes profile to agent via SystemMessage
      - update_profile tool updates the profile on disk when preferences are first mentioned

    Short-term memory:
      - history contains the full message history from previous turns (including ToolMessages)
      - This allows the agent to "see" the results of past searches in the next turn
      - Added to the query before calling the LLM

    Returns (response: str, updated_history: list).
    Hint: save ALL messages to history (HumanMessage, AIMessage, ToolMessage),
    so the agent knows what was found in the next turn.
    """
    updated_history = history.copy()
    updated_history.append(
        SystemMessage(content=f"User profile: {json.dumps(load_profile(profile_path), ensure_ascii=False)}")
    )
    updated_history.append(HumanMessage(content=user_message))

    while True:
        response = llm_chat(updated_history, tools=SHOP_TOOLS_SCHEMA_WITH_MEMORY)
        updated_history.append(response)

        if not response.tool_calls:
            return (response.content, updated_history)

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            if name == "search_products":
                result = tools.search_products(**args)
                state.last_results = result
            elif name == "add_to_cart":
                result = tools.add_to_cart(state, **args)
            elif name == "update_profile":
                result = update_profile(**args, path=profile_path)
            else:
                result = {"error": f"Unknown tool: {name}"}

            tracer.record(name=name, args=args, result=result)
            updated_history.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call_id))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3. Multi-Agent System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Implement a system of four agents + an orchestrator.
# Goal — find the best product and honestly describe its pros and cons.
# Agents work in a chain via a shared AgentContext object (defined in the template).
#
# RetrieverAgent (LLM + tools)
#   Searches for up to 5 relevant products via search_products.
#   Fills ctx.candidates and ctx.max_price.
#   Important: only pass the search tool (not add_to_cart).
#
# ProsAgent (LLM, no tools)
#   For each product in ctx.candidates, writes 1-2 sentences of pros.
#   Fills ctx.pros (dict: product_id -> pros string).
#   Records an "analyze_pros" call in tracer.
#
# ConsAgent (LLM, no tools)
#   For each product in ctx.candidates, writes 1-2 sentences of cons.
#   Fills ctx.cons (dict: product_id -> cons string).
#   Records an "analyze_cons" call in tracer.
#
# RankerAgent (no LLM — logic only)
#   Picks the best product from ctx.candidates:
#     - Filters by ctx.max_price (if set)
#     - Among remaining: highest rating; if tied — lowest price
#   Records a "rank_candidates" call in tracer. Fills ctx.best.
#
# CoordinatorAgent (orchestrator)
#   Runs agents in a chain, maintains a trace list.
#   Trace keys: "delegate_retriever", "delegate_pros", "delegate_cons",
#               "delegate_ranker", "delegate_cart".
#   No CartAgent needed — if the user asks to add to cart,
#   CoordinatorAgent does it itself via tools.add_to_cart after ranking.
#   Returns AgentResult with response, trace, and context.
#   The response should include: product name, price, rating, pros and cons.


@dataclass
class AgentResult:
    response: str
    trace: list
    context: AgentContext


class RetrieverAgent:
    def run(self, ctx: AgentContext, state: ShopState, tools: ShopTools, tracer: ToolTracer) -> AgentContext:
        """Searches for products via LLM+tools. Fills ctx.candidates and ctx.max_price."""
        search_schema = [convert_to_openai_tool(search_products)]
        system = (
            "You are a product retrieval assistant. "
            "Search for up to 5 products relevant to the user's query using the search_products tool. "
            "If the user mentions a maximum price, extract it and include it in your search filter."
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=ctx.query),
        ]

        while True:
            response = llm_chat(messages, tools=search_schema)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]

                result = tools.search_products(**args)
                state.last_results = result
                tracer.record(name=name, args=args, result=result)

                # Extract max_price from tool args if the LLM passed it
                if "max_price" in args and args["max_price"] is not None:
                    ctx.max_price = args["max_price"]

                # Accumulate candidates (deduplicated), cap at 5
                seen_ids = {p["id"] for p in ctx.candidates}
                for product in result:
                    if product["id"] not in seen_ids and len(ctx.candidates) < 5:
                        ctx.candidates.append(product)
                        seen_ids.add(product["id"])

                messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call_id))

        return ctx


class ProsAgent:
    def run(self, ctx: AgentContext, tracer: ToolTracer) -> AgentContext:
        """Finds pros for each product via LLM. Fills ctx.pros."""
        products_text = "\n".join(
            f"- id={p['id']} name={p['name']} brand={p['brand']} price={p['price']} rating={p['rating']} tags={p['tags']}"
            for p in ctx.candidates
        )
        messages = [
            SystemMessage(
                content="You are a product reviewer. "
                "For each product listed, write 1-2 sentences highlighting its main advantages. "
                "Reply with a JSON object mapping product id to pros string."
            ),
            HumanMessage(content=f"User query: {ctx.query}\n\nProducts:\n{products_text}"),
        ]
        response = llm_chat(messages)
        tracer.record(
            name="analyze_pros", args={"candidates": [p["id"] for p in ctx.candidates]}, result=response.content
        )

        try:
            parsed = json.loads(response.content)
            ctx.pros = parsed
        except json.JSONDecodeError:
            # Fallback: assign the whole response to each product
            for p in ctx.candidates:
                ctx.pros[p["id"]] = response.content

        return ctx


class ConsAgent:
    def run(self, ctx: AgentContext, tracer: ToolTracer) -> AgentContext:
        """Finds cons for each product via LLM. Fills ctx.cons."""
        products_text = "\n".join(
            f"- id={p['id']} name={p['name']} brand={p['brand']} price={p['price']} rating={p['rating']} tags={p['tags']}"
            for p in ctx.candidates
        )
        messages = [
            SystemMessage(
                content="You are a product reviewer. "
                "For each product listed, write 1-2 sentences highlighting its main disadvantages. "
                "Reply with a JSON object mapping product id to cons string."
            ),
            HumanMessage(content=f"User query: {ctx.query}\n\nProducts:\n{products_text}"),
        ]
        response = llm_chat(messages)
        tracer.record(
            name="analyze_cons", args={"candidates": [p["id"] for p in ctx.candidates]}, result=response.content
        )

        try:
            parsed = json.loads(response.content)
            ctx.cons = parsed
        except json.JSONDecodeError:
            for p in ctx.candidates:
                ctx.cons[p["id"]] = response.content

        return ctx


class RankerAgent:
    def run(self, ctx: AgentContext, tracer: ToolTracer) -> AgentContext:
        """Picks the best product from ctx.candidates considering ctx.max_price. Fills ctx.best."""
        candidates = ctx.candidates
        if ctx.max_price is not None:
            candidates = [p for p in candidates if p["price"] <= ctx.max_price]

        if candidates:
            ctx.best = max(candidates, key=lambda p: (p["rating"], -p["price"]))

        tracer.record(
            name="rank_candidates",
            args={"max_price": ctx.max_price, "candidates": [p["id"] for p in ctx.candidates]},
            result=ctx.best,
        )
        return ctx


class CoordinatorAgent:
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.pros_agent = ProsAgent()
        self.cons_agent = ConsAgent()
        self.ranker = RankerAgent()

    def run(self, user_message: str, state: ShopState, tools: ShopTools) -> AgentResult:
        """Orchestrates agents. Returns AgentResult with response, trace, and context."""
        tracer = ToolTracer()
        trace = []
        ctx = AgentContext(query=user_message)

        # Retriever
        self.retriever.run(ctx, state, tools, tracer)
        trace.append("delegate_retriever")

        # Pros & Cons (independent, but run sequentially for simplicity)
        self.pros_agent.run(ctx, tracer)
        trace.append("delegate_pros")

        self.cons_agent.run(ctx, tracer)
        trace.append("delegate_cons")

        # Ranker
        self.ranker.run(ctx, tracer)
        trace.append("delegate_ranker")

        # Add to cart if user asked for it and we have a best product
        if ctx.best and any(w in user_message.lower() for w in ("add", "buy", "cart")):
            result = tools.add_to_cart(state, ctx.best["id"])
            ctx.cart_result = result
            tracer.record(name="add_to_cart", args={"product_id": ctx.best["id"]}, result=result)
            trace.append("delegate_cart")

        # Build response
        if ctx.best is None:
            response = "Sorry, I couldn't find any products matching your request."
        else:
            b = ctx.best
            pros = ctx.pros.get(b["id"], "N/A")
            cons = ctx.cons.get(b["id"], "N/A")
            cart_note = ""
            if ctx.cart_result:
                cart_note = f"\n\nThe item has been added to your cart (cart size: {ctx.cart_result.get('cart_size')})."
            response = (
                f"**{b['name']}** by {b['brand']}\n"
                f"Price: ${b['price']} | Rating: {b['rating']}\n\n"
                f"**Pros:** {pros}\n"
                f"**Cons:** {cons}"
                f"{cart_note}"
            )

        return AgentResult(response=response, trace=trace, context=ctx)

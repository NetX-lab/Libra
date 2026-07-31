"""Support code for Search r1 prompt."""

# ------------------------------------------------------------------

# ------------------------------------------------------------------
SEARCH_R1_SYSTEM_PROMPT = "You are a helpful and harmless assistant."

# ------------------------------------------------------------------

# ------------------------------------------------------------------
SEARCH_R1_USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>.\n\nQuestion: "
)

# ------------------------------------------------------------------

# ------------------------------------------------------------------
TOOL_RESPONSE_TEMPLATE = "<tool_response>\n{results}\n</tool_response>"

# ------------------------------------------------------------------

# ------------------------------------------------------------------
SEARCH_R1_FORMAT_INSTRUCTION = (
    "Format your response as follows:\n"
    "1. First, reason inside <think>...</think> tags.\n"
    "2. If you need external knowledge, use <tool_call>your query</tool_call>.\n"
    "3. After you gather enough information, provide the final answer inside "
    "<answer>...</answer> tags.\n"
)

# ------------------------------------------------------------------

# ------------------------------------------------------------------
SEARCH_R1_FULL_SYSTEM = (
    SEARCH_R1_SYSTEM_PROMPT + "\n\n" + SEARCH_R1_FORMAT_INSTRUCTION
)

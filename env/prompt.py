COT_EXAMPLES = None
# COT_TASK_DESC = "<｜begin▁of▁sentence｜><｜User｜>\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜end▁of▁sentence｜>"
# PROBLEM_FORMAT_STR = """<｜begin▁of▁sentence｜><｜User｜>"\n{question}<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>\n"""
# COT_TASK_DESC = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"
# PROBLEM_FORMAT_STR = """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""
COT_TASK_DESC = "<｜begin▁of▁sentence｜><｜User｜>\nPlease reason step by step, and put your final answer within \\boxed{{}}"
PROBLEM_FORMAT_STR = """\n{question}<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>\n"""

SEP = "\n\n"

RM_STEP_TAG = "\n\n\n\n\n "
RM_FORMAT_STR = "{question} {answer}"

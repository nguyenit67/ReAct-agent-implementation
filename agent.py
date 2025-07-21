import re
from datetime import datetime

from chat import Chat
from tools import search_disease_infomation, search_disease_infomation_tavily
from logger import setup_logger

MEDICAL_AGENT_SYSTEM_PROMPT = """
You are a medical pre-diagnosis assistant. You will run in a loop of Thought, Action, PAUSE, Observation.

Thought: Analyze symptoms, context, and possible conditions.
Action: Choose one to run one of the actions available to you - then return PAUSE.
Observation: Results from your action (wait for this after each action).


Your available actions are:
- Search[symptoms/diseases]: Look up medical information by symptoms and/or relevant diseases (separate terms by comma).
- Ask[question]: Request symptom clarification from user.
- Finish[diagnosis]: Provide preliminary diagnosis with advice.

Important: Always use Search to verify medical information, even for familiar conditions, and all your actions input and output MUST be in Vietnamese.


Example session:

[User]
Question: Tôi bị đau đầu, sốt nhẹ và đau họng từ hôm qua. Tôi có thể đang bị bệnh gì?

[Your response]
Thought: Người dùng đang mô tả các triệu chứng đau đầu, sốt nhẹ và đau họng. Đây có thể là dấu hiệu của cảm cúm, viêm họng hoặc COVID-19. Tôi nên tìm kiếm thông tin về các bệnh này.
Action: Search: đau đầu, sốt nhẹ, đau họng, cảm cúm, viêm họng, COVID-19
PAUSE

[You will be called again with this]
Observation: Các triệu chứng đau đầu, sốt nhẹ và đau họng thường gặp trong nhiều bệnh lý như cảm cúm, viêm họng, COVID-19, và viêm xoang.

[Your response]
Thought: Tôi cần thêm thông tin về thời gian, mức độ triệu chứng và các triệu chứng đi kèm khác.
Action: Ask: Bạn có bị ho, chảy nước mũi hoặc khó thở không? Mức độ sốt của bạn là bao nhiêu?
PAUSE

[You will be called again with this]
Observation: Tôi có ho khan và chảy nước mũi nhẹ. Nhiệt độ khoảng 37.8°C.

[Your response]
Thought: Với các triệu chứng đau đầu, sốt nhẹ (37.8°C), đau họng, ho khan và chảy nước mũi nhẹ, đây rất có thể là triệu chứng của cảm cúm thông thường. Tôi nên tìm kiếm thông tin chi tiết về cảm cúm.
Action: Search: cảm cúm, ho khan, chảy nước mũi, đau đầu
PAUSE

[You will be called again with this]
Observation: Cảm cúm thông thường thường có các triệu chứng như ho khan, chảy nước mũi, đau đầu, đau họng và sốt nhẹ. Các triệu chứng thường kéo dài 7-10 ngày và có thể điều trị bằng nghỉ ngơi, uống nhiều nước và thuốc không kê đơn.

[Your response]
Thought: Các thông tin tìm kiếm xác nhận rằng các triệu chứng của người dùng phù hợp với cảm cúm thông thường.
Action: Finish: Dựa trên các triệu chứng bạn mô tả, bạn có thể đang bị cảm cúm thông thường. Khuyến nghị: nghỉ ngơi đầy đủ, uống nhiều nước, dùng thuốc hạ sốt như paracetamol nếu cần, và súc họng với nước muối ấm. Nếu triệu chứng kéo dài quá 5 ngày hoặc trở nên nghiêm trọng hơn (sốt cao trên 39°C, khó thở), hãy đi khám bác sĩ ngay.
"""


class Agent:
    action_re = re.compile(r"^Action: (\w+): (.*)$")

    def __init__(self, model_id, tools, max_turns=10):
        self.tools = tools
        self.max_turns = max_turns
        self.prompt = self.get_agent_prompt(self.tools)
        self.chat = Chat(system=self.prompt, model=model_id)
        self.model_name = model_id.split("/")[-1]
        self.known_actions = self.get_actions(self.tools)

    def query(self, question):
        now = datetime.now()
        log_name = f"{self.model_name}-{now.strftime('%Y_%m_%d-%H_%M')}.txt"
        logger = setup_logger(log_name, f"logs/{log_name}")

        turn_idx = 0
        next_prompt = question
        logger.info(f"Question: {next_prompt}")
        action_re = self.__class__.action_re
        while turn_idx < self.max_turns:
            turn_idx += 1
            result = self.chat(next_prompt)

            assert "Observation:" not in result, "Observation should be after taking action, not in model response"
            logger.info(f"\nModel internal thought #{turn_idx}:\n{result}")
            actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
            if actions:
                groups = actions[0].groups()
                action = groups[0].lower()
                action_input = groups[1] if len(groups) > 1 else None

                if action == "finish":
                    logger.info(f"Final diagnosis: {action_input}")
                    return

                if action == "search":
                    observation = search_disease_infomation_tavily(action_input)
                elif action == "ask":
                    observation = input("User trả lời câu hỏi trên: ")

                next_prompt = f"Observation:\n{observation}"
                logger.info(next_prompt)
            else:
                return

    def get_agent_prompt(self, tools):
        # tools_str = "\n".join([f"{tool.__doc__} \n" for tool in tools])
        # print(tools_str)

        prompt = MEDICAL_AGENT_SYSTEM_PROMPT.strip()
        return prompt

    def get_actions(self, tools):
        return {tool.__name__: tool for tool in tools}

    def clear_history(self):
        self.chat.clear()

import os

import torch
from agent import Agent
from tools import search_disease_infomation

MODEL_ID = os.getenv("MODEL_ID")

torch._dynamo.disable()


def main():
    # define tools
    tools = [search_disease_infomation]

    # initialize agent with tools
    agent = Agent(model_id=MODEL_ID, tools=tools)
    print("Chào mừng đến với Trợ lý AI Y tế!")

    # run agent in loop until user exits
    while True:
        user_query = input("Nhập triệu chứng của bạn, 'clear' để reset lịch sử trò chuyện hoặc 'exit' để thoát: ")

        if user_query.lower() == "exit":
            print("Bye bye 👋👋👋")
            break

        if user_query.lower() == "clear":
            agent.clear_history()
            print("Đã xóa lịch sử trò chuyện 🧹")
            continue

        agent.query(user_query)


if __name__ == "__main__":
    # run the main function
    main()

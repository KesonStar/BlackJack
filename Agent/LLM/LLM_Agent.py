import json
from openai import OpenAI

import sys

sys.path.append("../..")
from Game.BlackJack import BlackJack


class LLM:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.openai-proxy.org/v1",
            api_key="sk-VLyQzGVf1uvtxAS3nibssZLdXys2pK18Ys68XwJ2r5svG7Im",
        )
        self.messages = [
            {
                "role": "system",
                "content": "\
         现在在赌场内，我在玩一个名字叫做BlackJack的游戏，也叫做21点。\
         游戏内由我和PC两位玩家参与，游戏规则是数字牌（2到10）按照牌面的数字计算点数。\
         人头牌（J、Q、K）每张算作10点。A（Ace）算作11点。\
         游戏目标是尽可能让手中牌的总数到达21点。\
         一共两种gamemode，分别是对于player bust后的不同判断方式\
         traditional：Player bust后直接判lose\
         novel：Player bust后看Dealer，若dealer也bust则算draw\
            你可以选择hit或者stay，hit表示继续抽牌，stay表示停止抽牌。\
         你的输出代表我的决策，请尽可能让我获胜。\
         你的输出格式只能是hit或者stay。\
         ",
            }
        ]

    def choose_action(self, game: BlackJack) -> str:
        """
        chat 函数支持多轮对话，每次调用 chat 函数与 Kimi 大模型对话时，Kimi 大模型都会”看到“此前已经
        产生的历史对话消息，换句话说，Kimi 大模型拥有了记忆。
        """
        # 我们将用户最新的问题构造成一个 message（role=user），并添加到 messages 的尾部
        self.messages.append(
            {
                "role": "user",
                "content": f"我的牌是{game.player_hand}，我的点数是{game.get_playervalue()}。dealer的初始手牌之一是{game.format_cards(game.dealer_hand[:1])}。",
            }
        )

        # 携带 messages 与 Kimi 大模型对话
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",  # as for kimi: "moonshot-v1-8k",
            messages=self.messages,
            temperature=0.3,
        )

        # 通过 API 我们获得了 Kimi 大模型给予我们的回复消息（role=assistant）
        assistant_message = completion.choices[0].message

        # 为了让 Kimi 大模型拥有完整的记忆，我们必须将 Kimi 大模型返回给我们的消息也添加到 messages 中
        self.messages.append(assistant_message)
        print("I choose :", assistant_message.content)

        return (
            assistant_message.content
            if assistant_message.content in ["hit", "stay"]
            else exit("Invalid action")
        )

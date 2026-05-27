import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv(
    "DEEPSEEK_API_KEY"
)

# =========================
# 创建 DeepSeek 客户端
# =========================
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# =========================
# 自动定位项目根目录
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =========================
# 读取音乐数据库
# =========================
dataset_path = os.path.join(
    BASE_DIR,
    "database",
    "music_dataset.csv"
)

df = pd.read_csv(dataset_path)

print("DeepSeek 古典钢琴 AI 已启动")
print("输入 exit 退出")

# =========================
# AI问答循环
# =========================
while True:

    question = input("\n请输入问题： ")

    # 退出
    if question.lower() == "exit":
        print("系统已退出")
        break

    # 数据库统计
    stats = f'''
平均音高最高作曲家：
{df.groupby('canonical_composer')['avg_pitch'].mean().idxmax()}

音符最多作曲家：
{df.groupby('canonical_composer')['note_count'].mean().idxmax()}

最柔和作曲家：
{df.groupby('canonical_composer')['avg_duration'].mean().idxmax()}
'''

    # Prompt
    prompt = f'''
你是古典钢琴音乐分析AI。

下面是音乐数据库统计：

{stats}

用户问题：
{question}

请结合音乐知识与数据库结果，
给出专业、自然、简洁的回答。
'''

    # 调用 DeepSeek
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    # 获取回答
    answer = response.choices[0].message.content

    print("\nAI回答：\n")
    print(answer)


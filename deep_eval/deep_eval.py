from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric, HallucinationMetric
from custom_llm import Li_Custom_LLM
from custom_metric import Li_Custom_Metric

# ======================== GEval 一致性 ============================== #

# test_case = LLMTestCase(input="input to your LLM", actual_output="your LLM output")
# coherence_metric = GEval(
#     name="Coherence",
#     criteria="Coherence - the collective quality of all sentences in the actual output",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
#     model=Li_Custom_LLM()
# )

# coherence_metric.measure(test_case)
# print(coherence_metric.score, coherence_metric.reason)

# ======================== 答案相关性 ============================== #

# answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=Li_Custom_LLM())
# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the actual output from your LLM application
#     actual_output="We offer a 30-day full refund at no extra costs.",
#     # retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
# )

# answer_relevancy_metric.measure(test_case)
# print(answer_relevancy_metric.score)
# # All metrics also offer an explanation
# print(answer_relevancy_metric.reason)


# ======================== 幻觉评估 ============================== #

# 判断 LLM 输出是否符合背景上下文事实
context=["帖子内容：#理想L6提车9天后要换电机##理想汽车# 这波真的败好感，法律条文没写明，厂家就能推卸责任！？你们可以玩这种文字游戏，但中国电车品牌一定会受损，自己的路也会越走越窄！"]

# Replace this with the actual output from your LLM application
actual_output="评论：既然买理想，那就应该接受被喷，这是成为理想车主的代价。"

with open("./discriminator_prompt_neg.txt", "r", encoding="utf-8") as f:
    input_prompt = f.read()

test_case = LLMTestCase(
    input=input_prompt,
    actual_output=actual_output,
    context=context
)
metric = HallucinationMetric(threshold=0.5, model=Li_Custom_LLM())

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

result = evaluate(test_cases=[test_case], metrics=[metric])
print(result)


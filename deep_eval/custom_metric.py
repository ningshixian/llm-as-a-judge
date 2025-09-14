from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

# 强制 LLM 输出 json
# https://blog.csdn.net/jclian91/article/details/131446531
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List, Optional, Type, Union
import traceback

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.metrics.toxicity.template import ToxicityTemplate
from deepeval.metrics.toxicity.schema import *

"""使用自定义 Metric 进行评估
https://docs.confident-ai.com/guides/guides-building-custom-metrics#building-a-custom-composite-metric
"""

class evalScore(BaseModel):
    Positive: int = Field(description="评论的情感正向程度")
    Authenticity: int = Field(description="评论的真实程度")
    Relevance: int = Field(description="评论与帖子的相关程度")
    Personification: int = Field(description="评论的拟人程度")
    Diversity: int = Field(description="评论的互动性")

class evalReason(BaseModel):
    Positive: str = Field(description="评论情感正向的得分理由")
    Authenticity: str = Field(description="评论真实性的得分理由")
    Relevance: str = Field(description="评论与帖子相关性的得分理由")
    Personification: str = Field(description="评论拟人程度的得分理由")
    Diversity: str = Field(description="评论互动性的得分理由")

# 告诉他我们生成的内容需要哪些字段，每个字段类型式啥
class response_schemas(BaseModel):
    evalScore: evalScore
    evalReason: evalReason



class Li_Custom_Metric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,  # 0.5
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = False,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = self.get_prompt()

    def get_prompt(self):
        # 初始化解析器
        self.output_parser = PydanticOutputParser(pydantic_object=response_schemas)

        # 生成的格式提示符
        format_instructions = self.output_parser.get_format_instructions()
        # 处理Unicode字符时出现的中文乱码 https://www.cnblogs.com/Red-Sun/p/17219414.html
        # format_instructions = format_instructions.encode().decode('unicode_escape')
        # print(format_instructions)

        with open("./discriminator_prompt_neg.txt", "r", encoding="utf-8") as f:
            template = f.read()
            template += """\n% USER INPUT:\n{user_input}\nJSON输出："""
            # 创建PromptTemplate
            prompt = PromptTemplate(
                input_variables=["user_input"],
                partial_variables={"format_instructions": format_instructions},
                template=template
            )
        return prompt

    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[-1]
        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                # loop = get_or_create_event_loop()
                # loop.run_until_complete(
                #     self.a_measure(test_case, _show_indicator=False)
                # )
                pass
            else:
                self.verdicts: List[response_schemas] = self._generate_verdicts(
                    test_case.input, test_case.actual_output
                )

                self.score = self._calculate_score()
                self.reason = [self.verdicts[0].evalReason.__dict__.values()]
                self.success = self.score >= self.threshold

                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Reason:\n{prettify_list(self.reason)}",
                        f"Score: {self.score}",
                    ],
                )
            return self.score
    
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        pass
    
    def _generate_verdicts(self, input, actual_output) -> List[ToxicityVerdict]:

        verdicts: List[response_schemas] = []
        user_input = input + '\n' + actual_output
        prompt = self.evaluation_template.format(user_input=user_input)
        print(prompt)

        try:
            llm_output = self.model.generate(prompt, temperature=0.3)
            res = self.output_parser.parse(llm_output)
            print(res.evalScore.__dict__.values())
            print(res)
            print(res.model_dump_json(indent=4))  # 将 Pydantic 模型转换为 JSON 字符串
        except Exception as e:
            print(traceback.format_exc())
        
        verdicts = [res]
        return verdicts

    def _calculate_score(self) -> float:
        # 得分加权求和
        weights = [0.35, 0.1, 0.3, 0.15, 0.1]
        llm_score = sum(attr * weight for attr, weight in zip(self.verdicts[0].evalScore.__dict__.values(), weights))
        return llm_score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Li_Custom_Metric"


def test_llm():
    from custom_llm import Li_Custom_LLM
    evaluation_model = Li_Custom_LLM()

    metric = Li_Custom_Metric(
        threshold=80.0, 
        model=evaluation_model,
        verbose_mode=True
    )
    test_case = LLMTestCase(
        input="帖子内容：#理想L6提车9天后要换电机##理想汽车# 这波真的败好感，法律条文没写明，厂家就能推卸责任！？你们可以玩这种文字游戏，但中国电车品牌一定会受损，自己的路也会越走越窄！",
        # Replace this with the actual output from your LLM application
        actual_output="评论：既然买理想，那就应该接受被喷，这是成为理想车主的代价。",
        # expected_output="We offer a 30-day full refund at no extra costs.",
        # retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    # assert_test(test_case, [metric])

    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)
    print(metric.verdicts)


if __name__ == "__main__":
    #####################
    ### Example Usage ###
    #####################
    test_llm()

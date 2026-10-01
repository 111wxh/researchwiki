"""评测支线（PLAN §7）的数据与脚本包。

本包只依赖标准库与既有 wiki 包，零网络、零模型调用；
语料（fixtures）与题集（qa）存放在仓库根目录 ``evals/`` 下。
"""

from researchwiki.evals.qa import QaItem, load_qa, qa_type_counts

__all__ = ["QaItem", "load_qa", "qa_type_counts"]

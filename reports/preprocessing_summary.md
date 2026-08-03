# 最低限度预处理摘要

## 输入与删减

- 输入文件：`data/raw/Nazario_5.csv`
- 原始记录：3065
- 主题与正文同时为空而删除：0
- 精确重复组/涉及记录：15 / 30
- 删除的同标签精确重复：15
- 冲突标签重复组：0
- 最终记录：3050

## 固定拆分及类别数量

| split | legitimate (0) | phishing (1) | total | ratio |
|---|---:|---:|---:|---:|
| train | 1050 | 1094 | 2144 | 70.30% |
| validation | 225 | 229 | 454 | 14.89% |
| test | 223 | 229 | 452 | 14.82% |

## 8项特征范围

| feature | min | max |
|---|---:|---:|
| url_count | 0 | 652 |
| ip_address_url_count | 0 | 2 |
| urgency_word_count | 0 | 274 |
| credential_word_count | 0 | 1850 |
| action_word_count | 0 | 1311 |
| money_related_word_count | 0 | 597 |
| uppercase_letter_ratio | 0 | 1 |
| exclamation_mark_count | 0 | 177 |

## 关键检查

| check | result |
|---|---|
| 必需列存在且标签仅为0/1 | PASS |
| 8项特征均为有限数值 | PASS |
| 比例与计数特征范围合法 | PASS |
| email_id唯一且三个split互斥 | PASS |
| 每个split均包含两类邮件 | PASS |
| 固定种子重复运行结果一致 | PASS |

## 已知问题

- 保留了 3050 个现有冻结 split 分配；对原映射未覆盖的 0 封邮件按固定规则补充分配，没有重排既有记录。
- 为保留既有冻结映射，去重后的实际比例可能与70/15/15存在轻微取整偏差，具体比例见上表。
- 没有阻塞性未解决问题；测试集未用于关键词、特征、参数或阈值选择。

# CPEM 项目 Beamer 讲义（PPT 参考文稿）

文字密集型讲义，供制作课堂汇报或快闪 PPT 时**摘抄、删减、分页**。不是「一页一句」的极简演讲版。

## 编译

```bash
cd LatexCNBeamerTemplate/src
xelatex main.tex
xelatex main.tex
```

输出：`src/main.pdf`（约 15+ 页，含 allowframebreaks 自动分页）

## 章节结构

| 文件 | 内容 |
|------|------|
| `frontmatter.tex` | 封面、使用说明 |
| `section01.tex` | 项目介绍（背景、任务表、仓库结构） |
| `section02.tex` | 数据集 CPME（来源、划分、配置表） |
| `section03.tex` | 技术（情绪 CNN、MBTI TextCNN、训练、ONNX） |
| `section04.tex` | Web 功能、演示话术、局限对比 |
| `backmatter.tex` | 总结、Q&A、免责声明、PPT 页数映射 |

## 插图

- 论文 PDF：`pic/pipeline_overview.pdf`、`data_split.pdf` 等（自 `LatexCNArticleTemplate/pic/`）
- Material 插画：`pic/*.png`（由 `Material/*.svg` 经 `npx @resvg/resvg-js-cli` 生成）

## 制作自己的 PPT

1. 按 `backmatter` 末页「建议 PPT 页数映射」选取 frame  
2. 每页保留 3–5 条 bullet，删去 `\begin{block}` 内长段落  
3. **3 分钟快闪**：每节各取 1 页，约 6–7 页即可  

## 口播提纲（完整版约 8–10 分钟）

1. 封面 + 演示站  
2. 双模块目标与免责定位  
3. CPME 数据规模与用户级划分  
4. char-CNN 情绪 + TextCNN 注意力 MBTI  
5. 训练导出 ONNX、浏览器推理  
6. Web 四步演示 + 现场 checklist  
7. 局限与 Q&A  

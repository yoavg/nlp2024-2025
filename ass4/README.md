# Intro to NLP 2024-2025, Assignment 4.

**Prompting, question generation, and evaluation.**

## Introduction

In this assignment we will take a class we discussed in class (Question Generation), attempt to perform it using calls to an LLM, assess the results in various ways, and discuss our results.

## The Task

The task we attempt to perform is defined as follows:

> Given a short text and a span within the text, generate all the questions whose answer, based on the text, is the given span.
> The generated set of questions should be comprehensive (ask all the semantically different questions) but also distinct (if two questions in a set are just paraphrases of each other, we should be aware of it and treat them as the same question).
> The questions should be grammatically valid, fluent, and naturally sounding.

## The Data

You will work with a set of 100 text-span pairs that we extracted from the [SqUAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset. 

The data is in the following file:

- [data.jsonl](data.jsonl)

Each line in the file is a JSON object representing one text-span pair and several questions about it. The SqUAD data-collection was not meant to be exhaustive, so the set of questions may or may not be comprehensive.

In case you are interested or find this useful, the complete SqUAD training set, in HuggingFace format, is available [here](https://huggingface.co/datasets/rajpurkar/squad_v2). This set includes also the ones in the `data.jsonl` file above.

## Part 1: Manual Annotation

Select 20 text-span pairs from the file. Without looking at the provided questions, annotate (generate a comprehensive set of questions) for each of the 20 text-span pairs.

**Output of this part:**

- A file (`annotations.jsonl`) with your annotations in a JSON format to your choosing.
- Short discussion of your effort and reflection of the process, in the report.

## Part 2: Automatic Generation

Use an LLM API to perform the question generation task. For each of the text-span pairs in `data.jsonl`, you need to use the LLM to generate a good and comprehensive set of questions. Do your best to get high-quality results. Try and experiment with different prompting techniques, and different query formulations, as you deem relevant for the task.

**Output of this part:**

- A file (`generations.jsonl`) with generated questions, in the same JSON format as the previous part.
- A description of your final method, and discussion on why you chose it and how it compared to other methods you tried.


## Part 3: Basic Statistics

How many questions on average were generated for each pair? What is the distribution of number-of-questions over the entire set? How long on average is each question (choose a length metric that makes sense to you). Compare the numbers for your 

**Output of this part:**

- Readable graphs and/or tables with the above information.
- A brief discussion of the results and what you learn from them.

Both of these items should be incorporated in your report.

## Part 4: Automatic Evaluation via the Rouge metric

@@TBD BY REUT/IDO@@

**Output of this part:**

@@TBD BY REUT/IDO@@

## Part 5: Validation using an LLM

A common practice is to use a process that involves an LLM as a **validator**. Here, we check the quality of the generated questions based on the LLM's ability to handle them. 

In a "real-life" process, usually this part acts as quality assurance (if many questions are not handled well by the LLM, maybe the entire process is bad), or, more commonly, as a filter (if only a small number of questions are not handled well, maybe we should just throw them---or the pairs they are in---out of the set). Another use is to build a more elaborate process that attempts to improve the items the LLM validator failed on. **In this work** we will just perform the validation and record the result (i.e., we will count, but will not filter).

Concretely, you need to validate the ability of the LLM to answer the generated questions, based on the text, and get the same (or equivalent) answers  to the original spans. Do this by writing a strong prompt template that will receive instructions, text and question, and answer the question based on the text. Run this on all the generated questions, record the answers, and validate them.

A related quantity we ask you to compute is the ability of the model to answer the questions _without seeing the texts_ while providing the same (or equivalent) asnwers as those based on the texts.

**Output of this part:**

- A description of your procedure / prompts.
- The LLM's output (`validation_outputs.jsonl`).
- Readable graphs and/or tables with the above information.
- A brief discussion of the results and what you learn from them.

Your JSON format should include all the requested items, in a way that is understandable and clear to an external reader.

## Part 5: Evaluation using an LLM ("LLM as a judge")

Another way of using LLM is to use them directly as evaluators of quality. Here, we wish the LLM to directly answer two questions related to the quality of the questions:

1) Is the question grammatical and fluent?

2) Is the question answerable from the text, and if so, is the answer according to the text consistent with the original intended answer?

Note that question (2) relies on the same information the LLM verification process was intended to measure, but here we ask the LLM to assess it directly, while in the validation case we ask the LLM to perform the task and interpret the results --- a different computational process.

It is interesting to note if these processes agree or disagree with each other, and the implications of this.

**Output of this part:**

- A description of your procedure / prompts.
- The LLM's output. (`llm_judge_outputs.jsonl`)
- Readable graphs and/or tables with the above information (evaluation of the generated questions according to each metric).
- A brief discussion of the results and what you learn from them.

Your JSON format should include all the requested items, in a way that is understandable and clear to an external reader.

## Part 6: Manual Evaluation

Finally, we want you to perform manual evaluation. Use the 20 text-span pairs you fully annotated.

For each one, assess the different questions generated by the LLM.

First, for each question, use the exact two criteria we asked the LLM above:

1) Is the question grammatical and fluent?

2) Is the question answerable from the text, and if so, is the answer according to the text consistent with the original intended answer?

We also ask you to answer another criteria:

3) Is the question "good" (in your judgement)

Finally, we want you to evaluate the generated _set_ of questions: 

4) For each of the 20 text-span pairs, is the _set of questions_ exhaustive? Compare it to your gold annotation, are all questions covered? Are there questions in one set and not in the other? Are there repeated questions? Etc.

**Output of this part:**

- A description of your procedure.
- Readable graphs and/or tables with the above information (evaluation of the generated questions according to each metric).
- A brief discussion of the results and what you learn from them.

## Part 7: Wrapping it up

Write a summary of the entire process and the overall results. What did you learn from it? Focus on the correlations between the different metrics and techniques, and what we can learn from them. Discuss also the strengths and weaknesses of the different approaches, specific challenges you found, etc.

Include also a discussion of the idea to use an LLM to evaluate the outputs of the same LLM. What are your general thoughts about the idea? Is it a good idea in general? If so, why, if no, why not. Is it a good idea sometimes? If so, when, and when not. What can it measure, and what can’t it measure? Can it be used as a replacement for human evaluation? If so, when. If not, why not. Etc.


**Output of this part:**

- The concluding chapter of your report.

## What to submit:

- **A single report file in PDF format**, including your answers (tables, graphs, methods, discussions) from the different parts. Include your names and IDs clearly at the top of the report file. Make it clear which part of your report answers which section, and make clear what each graph/table/answer/... in the report represents or answers.
- The following files:
    - `annotations.jsonl`
    - `generations.jsonl`
    - ROUGE FILES TBD
    - `validation_outputs.jsonl`
    - `llm_judge_outputs.jsonl`

As in the previous assignments, a large part of your grade will be based on your report / essay.








